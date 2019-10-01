#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include "activation_layer.h"
#include "logistic_layer.h"
#include "l2norm_layer.h"
#include "activations.h"
#include "avgpool_layer.h"
#include "batchnorm_layer.h"
#include "blas.h"
#include "connected_layer.h"
#include "deconvolutional_layer.h"
#include "convolutional_layer.h"
#include "cost_layer.h"
#include "crnn_layer.h"
#include "crop_layer.h"
#include "detection_layer.h"
#include "dropout_layer.h"
#include "gru_layer.h"
#include "list.h"
#include "local_layer.h"
#include "maxpool_layer.h"
#include "normalization_layer.h"
#include "option_list.h"
#include "parser.h"
#include "region_layer.h"
#include "yolo_layer.h"
#include "iseg_layer.h"
#include "reorg_layer.h"
#include "rnn_layer.h"
#include "route_layer.h"
#include "upsample_layer.h"
#include "shortcut_layer.h"
#include "softmax_layer.h"
#include "lstm_layer.h"
#include "utils.h"

#include "darknet.h"

extern LAYER_TYPE string_to_layer_type(char * type);

layer mktest_convolutional() {
	layer l={0};

 	return(l);
}

layer mktest_deconvolutional() {
	layer l={0};

 	return(l);
}

layer mktest_local() {
	layer l={0};

 	return(l);
}

layer mktest_activation() {
	layer l={0};

 	return(l);
}

layer mktest_logistic() {
	layer l={0};

 	return(l);
}

layer mktest_l2norm() {
	layer l={0};

 	return(l);
}

layer mktest_rnn() {
	layer l={0};

 	return(l);
}

layer mktest_gru() {
	layer l={0};

 	return(l);
}

layer mktest_lstm() {
	layer l={0};

 	return(l);
}

layer mktest_crnn() {
	layer l={0};

 	return(l);
}

layer mktest_connected() {
	layer l={0};

 	return(l);
}

layer mktest_crop() {
	layer l={0};

 	return(l);
}

layer mktest_cost() {
	layer l={0};

 	return(l);
}

layer mktest_region() {
	layer l={0};

 	return(l);
}

layer mktest_yolo() {
	layer l={0};

 	return(l);
}

layer mktest_iseg() {
	layer l={0};

 	return(l);
}

layer mktest_detection() {
	layer l={0};

 	return(l);
}

layer mktest_softmax() {
	layer l={0};

 	return(l);
}

layer mktest_normalization() {
	layer l={0};

 	return(l);
}

layer mktest_batchnorm() {
	layer l={0};

 	return(l);
}

layer mktest_maxpool() {
	layer l={0};

 	return(l);
}

layer mktest_reorg() {
	layer l={0};

 	return(l);
}

layer mktest_avgpool() {
	layer l={0};

 	return(l);
}

layer mktest_route() {
	layer l={0};

 	return(l);
}

layer mktest_upsample() {
	layer l={0};

 	return(l);
}

layer mktest_shortcut() {
	layer l={0};

 	return(l);
}

layer build_network_cfg(char *layername)
{
    layer l;
    fprintf(stderr, "layer     filters    size              input                output\n");
    do { 
       LAYER_TYPE lt = string_to_layer_type(layername);

       switch(lt) {

        case CONVOLUTIONAL:
            l = mktest_convolutional();
			break;

        case DECONVOLUTIONAL:
            l = mktest_deconvolutional();
			break;

        case LOCAL:
            l = mktest_local();
			break;

        case ACTIVE:
            l = mktest_activation();
			break;

        case LOGXENT:
            l = mktest_logistic();
			break;

        case L2NORM:
            l = mktest_l2norm();
			break;

        case RNN:
            l = mktest_rnn();
			break;

        case GRU:
            l = mktest_gru();
			break;

        case LSTM:
            l = mktest_lstm();
			break;

        case CRNN:
            l = mktest_crnn();
			break;

        case CONNECTED:
            l = mktest_connected();
			break;

        case CROP:
            l = mktest_crop();
			break;

        case COST:
            l = mktest_cost();
			break;

        case REGION:
            l = mktest_region();
			break;

        case YOLO:
            l = mktest_yolo();
			break;

        case ISEG:
            l = mktest_iseg();
			break;

        case DETECTION:
            l = mktest_detection();
			break;

        case SOFTMAX:
            l = mktest_softmax();
			break;

        case NORMALIZATION:
            l = mktest_normalization();
			break;

        case BATCHNORM:
            l = mktest_batchnorm();
			break;

        case MAXPOOL:
            l = mktest_maxpool();
			break;

        case REORG:
            l = mktest_reorg();
			break;

        case AVGPOOL:
            l = mktest_avgpool();
			break;

        case ROUTE:
            l = mktest_route();
			break;

        case UPSAMPLE:
            l = mktest_upsample();
			break;

        case SHORTCUT:
            l = mktest_shortcut();
			break;

        case DROPOUT:
            break;

        default:
            printf("Unknown layer\n");
			break;

       }
    } while(0);
    
    return l;
}

int main(int argc, char **argv) {
   
    if(argc < 2){
        fprintf(stderr, "usage: %s [layer_type] -i [gpuIndex]\n", argv[0]);
        return 0;
    }
    gpu_index = find_int_arg(argc, argv, "-i", 0);
    if(find_arg(argc, argv, "-nogpu")) {
        gpu_index = -1;
    }

    printf("layer type %d\n",(int)string_to_layer_type(argv[1]));
    
    exit(0);    
}

