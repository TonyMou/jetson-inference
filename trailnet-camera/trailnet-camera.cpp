/*
 * http://github.com/dusty-nv/jetson-inference
 */

#include "gstCamera.h"

#include "glDisplay.h"
#include "glTexture.h"

#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <math.h>

#include "cudaMappedMemory.h"
#include "cudaNormalize.h"
#include "cudaFont.h"

#include "trailNet.h"


#define DEFAULT_CAMERA -1	// -1 for onboard camera, or change to index of /dev/video V4L2 camera (>=0)	
		
static int   dnn_class_count_       = 6;       // how many dnn classes are used for control

static float dnn_turn_angle_ = 10.f; // how much dnn turns each time to keep orientation (in degrees)
static float turn_angle_     = 0.f; // used for filtering
static float dnn_lateralcorr_angle_ = 10.f; // how much dnn turns each time to keep middle of the path position (in degrees)
static float direction_filter_innov_coeff_; // 0..1.0f how much of new control to integrate to the current command

static float linear_control_val_   = 0;  // forward control: "+" is forward and "-" is back (-1..1), updated in the JOYSTICK subsriber
static float angular_control_val_  = 0; // turn control: "+" turns left and "-" turns right (-1..1), updated in the JOYSTICK subsriber
static float yaw_control_val_      = 0;     // yaw control: "+" rotates in place to left and "-" rotates right (-1..1)
static float altitude_control_val_ = 0; // altitude control: "+" is up and "-" is down (-1..1)

// DNN controls are used when enabled and when joystick is not in use
static bool  use_dnn_data_            = false; // whether to use data from DNN or not.
static bool  got_new_dnn_command_     = false; // whether we've got a new command from DNN
static float dnn_linear_control_val_  = 0 ;    // dnn forward control: "+" is forward and "-" is back (-1..1), updated in the DNN subsriber
static float dnn_angular_control_val_ = 0;     // dnn turn control: "+" turns left and "-" turns right (-1..1), updated in the DNN subsriber

bool signal_recieved = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		printf("received SIGINT\n");
		signal_recieved = true;
	}
}

float from_degrees(float degrees)
{
	return degrees * M_PI / 180.0f;
}

float to_degrees(float radians)
{
	return radians * 180.0f / M_PI;
}

void computeDNNControl(const float class_probabilities[6], float& linear_control_val, float& angular_control_val)
{
    // Normalize probabilities just in case. We have 6 classes, they are disjoint - 1st 3 are rotations and 2nd 3 are translations
    float prob_sum = class_probabilities[0] + class_probabilities[1] + class_probabilities[2];
    if (prob_sum < 1e-6)return;
    //assert(prob_sum!=0);
    float left_view_p   = class_probabilities[0] / prob_sum;
    float right_view_p  = class_probabilities[2] / prob_sum;

    prob_sum = class_probabilities[3] + class_probabilities[4] + class_probabilities[5];
    if (prob_sum < 1e-6)return;
    // assert(prob_sum!=0);
    float left_side_p   = class_probabilities[3] / prob_sum;
    float right_side_p  = class_probabilities[5] / prob_sum;

    // Compute turn angle from probabilities. Positive angle - turn left, negative - turn right, 0 - go straight
    float current_turn_angle_deg =  dnn_turn_angle_*(right_view_p - left_view_p) + dnn_lateralcorr_angle_*(right_side_p - left_side_p);

    // Do sanity check and convert to radians
    current_turn_angle_deg = std::max(-90.0f, std::min(current_turn_angle_deg, 90.0f));   // just in case to avoid bad control
    //bugs
    float current_turn_angle_rad = from_degrees((float)current_turn_angle_deg);

    // Filter computed turning angle with the exponential filter
    turn_angle_ = turn_angle_*(1-direction_filter_innov_coeff_) + current_turn_angle_rad*direction_filter_innov_coeff_; // TODO: should this protected by a lock?
    float turn_angle_rad = turn_angle_;
    // end of turning angle filtering

    printf("DNN turn angle: %4.2f deg.", to_degrees(turn_angle_rad));
    // Create control values that lie on a unit circle to mimic max joystick control values that are on a unit circle
    //bugs
    linear_control_val  = cosf(turn_angle_rad);
    angular_control_val = sinf(turn_angle_rad);
}

int main( int argc, char** argv )
{
	printf("trailnet-camera\n  args (%i):  ", argc);

	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);
		
	printf("\n\n");

	direction_filter_innov_coeff_ = 1.0f;
	dnn_lateralcorr_angle_ = 10.0f;
	dnn_turn_angle_ = 10.f;
	turn_angle_ = 0.f;
	

	/*
	 * parse network type from CLI arguments
	 */
	trailNet::NetworkType networkType = trailNet::TRAILNET;
/*
	if( argc > 1 )
	{
		if( strcmp(argv[1], "trailnet") == 0 )
			networkType = trailNet::TRAILNET;
		else if( strcmp(argv[1], "ped-100") == 0 )
			networkType = detectNet::PEDNET;
		else if( strcmp(argv[1], "facenet") == 0 || strcmp(argv[1], "facenet-120") == 0 || strcmp(argv[1], "face-120") == 0 )
			networkType = detectNet::FACENET;
	}*/
	
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("\ncan't catch SIGINT\n");


	/*
	 * create the camera device
	 */
	gstCamera* camera = gstCamera::Create(DEFAULT_CAMERA);
	
	if( !camera )
	{
		printf("\ntrailnet-camera:  failed to initialize video device\n");
		return 0;
	}
	
	printf("\ntrailnet-camera:  successfully initialized video device\n");
	printf("    width:  %u\n", camera->GetWidth());
	printf("   height:  %u\n", camera->GetHeight());
	printf("    depth:  %u (bpp)\n\n", camera->GetPixelDepth());
	

	/*
	 * create detectNet
	 */
	trailNet* net = trailNet::Create(networkType);
	
	if( !net )
	{
		printf("trailnet-camera:   failed to initialize imageNet\n");
		return 0;
	}


	
	float* probCPU    = NULL;
	float* probCUDA   = NULL;
	
	if( !cudaAllocMapped((void**)&probCPU, (void**)&probCUDA, 6 * sizeof(float)))
	{
		printf("trailnet-console:  failed to alloc output memory\n");
		return 0;
	}
	

	/*
	 * create openGL window
	 */
	glDisplay* display = glDisplay::Create();
	glTexture* texture = NULL;
	
	if( !display ) {
		printf("\ntrailnet-camera:  failed to create openGL display\n");
	}
	else
	{
		texture = glTexture::Create(camera->GetWidth(), camera->GetHeight(), GL_RGBA32F_ARB/*GL_RGBA8*/);

		if( !texture )
			printf("trailnet-camera:  failed to create openGL texture\n");
	}
	
	
	/*
	 * create font
	 */
	cudaFont* font = cudaFont::Create();
	

	/*
	 * start streaming
	 */
	if( !camera->Open() )
	{
		printf("\ntrailnet-camera:  failed to open camera for streaming\n");
		return 0;
	}
	
	printf("\ntrailnet-camera:  camera open for streaming\n");
	
	
	/*
	 * processing loop
	 */
	float confidence = 0.0f;
	float class_probabilities[6] = { 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f };
	while( !signal_recieved )
	{
		void* imgCPU  = NULL;
		void* imgCUDA = NULL;
		
		// get the latest frame
		if( !camera->Capture(&imgCPU, &imgCUDA, 1000) )
			printf("\ntrailnet-camera:  failed to capture frame\n");

		// convert from YUV to RGBA
		void* imgRGBA = NULL;
		
		if( !camera->ConvertRGBA(imgCUDA, &imgRGBA) )
			printf("trailnet-camera:  failed to convert from NV12 to RGBA\n");

		// classify image with detectNet
		int numProbs = 6;
	
		if( net->forward((float*)imgRGBA, camera->GetWidth(), camera->GetHeight(), 3, probCPU))
		{
		
			//compute dnn control
			// bug	
			class_probabilities[0] = probCPU[0];
			class_probabilities[1] = probCPU[1];
			class_probabilities[2] = probCPU[2];
			class_probabilities[3] = 0.0f;
			class_probabilities[4] = 1.0f;
			class_probabilities[5] = 0.0f;

		    if(dnn_class_count_ == 6)
		    {
		        class_probabilities[3] = probCPU[3];
		        class_probabilities[4] = probCPU[4];
		        class_probabilities[5] = probCPU[5];
		    }
		    // bugs
			printf("DNN state/message: on=%d, %4.2f, %4.2f, %4.2f, %4.2f, %4.2f, %4.2f\n", (int)use_dnn_data_,
		    	class_probabilities[0], class_probabilities[1], class_probabilities[2],
		    	class_probabilities[3], class_probabilities[4], class_probabilities[5]);

		    computeDNNControl(class_probabilities, dnn_linear_control_val_, dnn_angular_control_val_);

			/*if( font != NULL )
			{
				char str[256];
				sprintf(str, "%05.2f%% %s", confidence * 100.0f, net->GetClassDesc(img_class));
				
				font->RenderOverlay((float4*)imgRGBA, (float4*)imgRGBA, camera->GetWidth(), camera->GetHeight(),
								    str, 10, 10, make_float4(255.0f, 255.0f, 255.0f, 255.0f));
			}*/
			
			if( display != NULL )
			{
				char str[256];
				sprintf(str, "TensorRT build %x | %s | %04.1f FPS", NV_GIE_VERSION, net->HasFP16() ? "FP16" : "FP32", display->GetFPS());
				//sprintf(str, "GIE build %x | %s | %04.1f FPS | %05.2f%% %s", NV_GIE_VERSION, net->GetNetworkName(), display->GetFPS(), confidence * 100.0f, net->GetClassDesc(img_class));
				display->SetTitle(str);	
			}	
		}	


		// update display
		if( display != NULL )
		{
			display->UserEvents();
			display->BeginRender();

			if( texture != NULL )
			{
				// rescale image pixel intensities for display
				CUDA(cudaNormalizeRGBA((float4*)imgRGBA, make_float2(0.0f, 255.0f), 
								   (float4*)imgRGBA, make_float2(0.0f, 1.0f), 
		 						   camera->GetWidth(), camera->GetHeight()));

				// map from CUDA to openGL using GL interop
				void* tex_map = texture->MapCUDA();

				if( tex_map != NULL )
				{
					cudaMemcpy(tex_map, imgRGBA, texture->GetSize(), cudaMemcpyDeviceToDevice);
					texture->Unmap();
				}

				// draw the texture
				texture->Render(100,100);		
			}

			display->EndRender();
		}
	}
	
	printf("\ntrailnet-camera:  un-initializing video device\n");
	
	
	/*
	 * shutdown the camera device
	 */
	if( camera != NULL )
	{
		delete camera;
		camera = NULL;
	}

	if( display != NULL )
	{
		delete display;
		display = NULL;
	}
	
	printf("trailnet-camera:  video device has been un-initialized.\n");
	printf("trailnet-camera:  this concludes the test of the video device.\n");
	return 0;
}

