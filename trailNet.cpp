#include "trailNet.h"
#include "cudaMappedMemory.h"
#include "cudaResize.h"
#include "commandLine.h"

// constructor
trailNet::trailNet() : tensorNet()
{
	
}


// destructor
trailNet::~trailNet()
{

}


// Create
trailNet* trailNet::Create(trailNet::NetworkType networkType, uint32_t maxBatchSize )
{
	trailNet* net = new trailNet();
	
	if( !net )
		return NULL;
	
	if( !net->init(networkType, maxBatchSize) )
	{
		printf("imageNet -- failed to initialize.\n");
		return NULL;
	}
	
	return net;
}


// Create
trailNet* trailNet::Create( const char* prototxt_path, const char* model_path, const char* input, const char* output, uint32_t maxBatchSize )
{
	trailNet* net = new trailNet();
	
	if( !net )
		return NULL;
	
	if( !net->init(prototxt_path, model_path, input, output, maxBatchSize) )
	{
		printf("imageNet -- failed to initialize.\n");
		return NULL;
	}
	
	return net;
}


// init
bool trailNet::init(NetworkType networkType, uint32_t maxBatchSize )
{
	return init( "networks/TrailNet/TrailNet_SResNet-18.prototxt", "networks/TrailNet/TrailNet_SResNet-18.caffemodel", TRAILNET_DEFAULT_INPUT, TRAILNET_DEFAULT_OUTPUT, maxBatchSize );
}


// init
bool trailNet::init(const char* prototxt_path, const char* model_path, const char* input, const char* output, uint32_t maxBatchSize )
{
	if( !prototxt_path || !model_path || !input || !output )
		return false;

	printf("\n");
	printf("imageNet -- loading classification network model from:\n");
	printf("         -- prototxt     %s\n", prototxt_path);
	printf("         -- model        %s\n", model_path);
	printf("         -- input_blob   '%s'\n", input);
	printf("         -- output_blob  '%s'\n", output);
	printf("         -- batch_size   %u\n\n", maxBatchSize);

	/*
	 * load and parse trailnet network definition and model file
	 */
	if( !tensorNet::LoadNetwork( prototxt_path, model_path, NULL, input, output, maxBatchSize ) )
	{
		printf("failed to load %s\n", model_path);
		return false;
	}

	printf(LOG_GIE "%s loaded\n", model_path);
	
	printf("%s initialized.\n", model_path);
	return true;
}
			


// Create
trailNet* trailNet::Create( int argc, char** argv )
{
	commandLine cmdLine(argc, argv);

	const char* modelName = cmdLine.GetString("model");

	if( !modelName )
	{
		if( argc == 2 )
			modelName = argv[1];
		else if( argc == 4 )
			modelName = argv[3];
		else
			modelName = "trailnet";
	}

	//if( argc > 3 )
	//	modelName = argv[3];
	trailNet::NetworkType type = trailNet::TRAILNET;


	if( strcasecmp(modelName, "trailnet") == 0 )
	{
		type = trailNet::TRAILNET;
	}
	else
	{
		const char* prototxt = cmdLine.GetString("prototxt");
		const char* labels   = cmdLine.GetString("labels");
		const char* input    = cmdLine.GetString("input_blob");
		const char* output   = cmdLine.GetString("output_blob");
		const char* out_bbox = cmdLine.GetString("output_bbox");
		
		if( !input ) 	input    = TRAILNET_DEFAULT_INPUT;
		if( !output )  output   = TRAILNET_DEFAULT_OUTPUT;
		
		int maxBatchSize = cmdLine.GetInt("batch_size");
		
		if( maxBatchSize < 1 )
			maxBatchSize = 2;

		return trailNet::Create(prototxt, modelName, input, output, maxBatchSize);
	}

	// create from pretrained model
	return trailNet::Create(type);
}


cudaError_t cudaPreImageNetMean( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float3& mean_value );

				 
bool trailNet::forward(const float* rgba, size_t width, size_t height, size_t channel, float* probs)
{
	if( !rgba || width == 0 || height == 0)
	{
		printf("trailNet::Detect( 0x%p, %u, %u ) -> invalid parameters\n", rgba, width, height);
		return false;
	}

	
	// downsample and convert to band-sequential BGR
	if( CUDA_FAILED(cudaPreImageNetMean((float4*)rgba, width, height, mInputCUDA, mWidth, mHeight,
								  make_float3(104.0069879317889f, 116.66876761696767f, 122.6789143406786f))) )
	{
		printf("trailNet::Classify() -- cudaPreImageNetMean failed\n");
		return false;
	}
	
	// process with GIE
	void* inferenceBuffers[] = { mInputCUDA, mOutputs[0].CUDA };
	
	if( !mContext->execute(1, inferenceBuffers) )
	{
		printf(LOG_GIE "trailNet::Classify() -- failed to execute tensorRT context\n");
		
		return false;
	}
	
	PROFILER_REPORT();

	// cluster detection bboxes
	float* output   = mOutputs[0].CPU;
	
	const int ow  = DIMS_W(mOutputs[0].dims);		// number of columns in bbox grid in X dimension
	const int oh  = DIMS_H(mOutputs[0].dims);		// number of rows in bbox grid in Y dimension
	const int oc  = DIMS_C(mOutputs[0].dims);
	
	const int osize = ow * oh * oc;
	
	memcpy(probs, output, osize * sizeof(float));
	
	return true;
}
