/*
 * zerotech.com
 */
 
#ifndef __TRAIL_NET_H__
#define __TRAIL_NET_H__


#include "tensorNet.h"


/**
 * Name of default input blob for trailNet model.
 * @ingroup deepVision
 */
#define TRAILNET_DEFAULT_INPUT   "data"

/**
 * Name of default output confidence values for trailNet model.
 * @ingroup deepVision
 */
#define TRAILNET_DEFAULT_OUTPUT  "out"


/**
 * Image recognition with trailNet, using TensorRT.
 * @ingroup deepVision
 */
class trailNet : public tensorNet
{
public:
	/**
	 * Network choice enumeration.
	 */
	enum NetworkType
	{
		TRAILNET,
		OTHERS
	};
	/**
	 * Load a new network instance
	 */
	static trailNet* Create(NetworkType networkType=TRAILNET,uint32_t maxBatchSize=2 );
	
	/**
	 * Load a new network instance
	 * @param prototxt_path File path to the deployable network prototxt
	 * @param model_path File path to the caffemodel
	 * @param mean_binary File path to the mean value binary proto (can be NULL)
	 * @param class_info File path to list of class name labels
	 * @param input Name of the input layer blob.
	 * @param maxBatchSize The maximum batch size that the network will support and be optimized for.
	 */
	static trailNet* Create( const char* prototxt_path, const char* model_path, 
						const char* input=TRAILNET_DEFAULT_INPUT, 
						const char* output=TRAILNET_DEFAULT_OUTPUT, 
						uint32_t maxBatchSize=2 );
	
	/**
	 * Load a new network instance by parsing the command line.
	 */
	static trailNet* Create( int argc, char** argv );

	/**
	 * Destroy
	 */
	virtual ~trailNet();
	
	/**
	 * Determine the maximum likelihood image class.
	 * @param rgba float4 input image in CUDA device memory.
	 * @param width width of the input image in pixels.
	 * @param height height of the input image in pixels.
	 * @param confidence optional pointer to float filled with confidence value.
	 * @returns Index of the maximum class, or -1 on error.
	 */
	bool forward(const float* input, size_t w, size_t h, size_t c, float *probs);

protected:
	trailNet();
	
	bool init( NetworkType networkType, uint32_t maxBatchSize );
	bool init(const char* prototxt_path, const char* model_path, const char* input, const char* output, uint32_t maxBatchSize );
	
};


#endif

