#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

#include <stdio.h>
#include <unistd.h>

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

#define Calloc(n, t)   (t *) calloc( (size_t) (n), sizeof(t) )
#define asizeof(a)     (int)(sizeof (a) / sizeof ((a)[0]))

extern LAYER_TYPE string_to_layer_type(char * type);

layer mktest_convolutional(int argc, char **argv) {
	layer l = {0};
	
    if( argc < 3 )
		printf("\tmake_convolutional_layer(batch,h,w,c,n,groups,size,stride,padding,activation, batch_normalize, binary, xnor, net->adam)\n\n;");

 	return(l);
}

layer mktest_deconvolutional(int argc, char **argv) {
	layer l = {0};
	
      if( argc < 3 )
		printf("\tmake_deconvolutional_layer(batch,h,w,c,n,size,stride,padding, activation, batch_normalize, net->adam)\n\n");

 	return(l);
}

layer mktest_local(int argc, char **argv) {
	layer l = {0};
	
    if( argc < 3 )
		printf("local_make_local_layer(batch,h,w,c,n,size,stride,pad,activation)\n\n");

 	return(l);
}

layer mktest_activation(int argc, char **argv) {

    int batch = find_int_arg(argc, argv, "-b", 100);
    int inputs = find_int_arg(argc, argv, "-i", 2000);
    int activation = find_int_arg(argc, argv, "-a", 10) ;

    if( argc < 3 )
		printf("	make_activation_layer(batch, inputs, activation[0..5])\n"
            "LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN, SELU\n\n");
    layer l = make_activation_layer(batch, inputs, activation);

 	return(l);
}

layer mktest_logistic(int argc, char **argv) {

    int batch = find_int_arg(argc, argv, "-b", 100);
    int inputs = find_int_arg(argc, argv, "-i", 2000);
	
    if( argc < 3 )
		printf("	make_logistic_layer(batch, inputs)\n\n");

    layer l = make_logistic_layer(batch, inputs);

 	return(l);
}

layer mktest_l2norm(int argc, char **argv) {
	
    int batch = find_int_arg(argc, argv, "-b", 100);
    int inputs = find_int_arg(argc, argv, "-i", 2000);
	
    if( argc < 3 )
		printf("	make_l2norm_layer(batch, inputs)\n\n");

    layer l = make_l2norm_layer(batch, inputs);

 	return(l);
}

layer mktest_rnn(int argc, char **argv) {
	layer l = {0};
	
      if( argc < 3 )
		printf("	make_rnn_layer(batch, inputs, output, time_steps, activation, batch_normalize, net->adam)\n\n");

 	return(l);
}

layer mktest_gru(int argc, char **argv) {
	layer l = {0};
	
      if( argc < 3 )
		printf("	make_gru_layer(batch, inputs, output, time_steps, batch_normalize, net->adam)\n\n");

 	return(l);
}

layer mktest_lstm(int argc, char **argv) {
	layer l = {0};
	
      if( argc < 3 )
		printf("	make_lstm_layer(batch, inputs, output, time_steps, batch_normalize, net->adam)\n\n");

 	return(l);
}

layer mktest_crnn(int argc, char **argv) {
	layer l = {0};
	
      if( argc < 3 )
		printf("	make_crnn_layer(batch, w, h, c, hidden_filters, output_filters, time_steps, activation, batch_normalize)\n\n");

 	return(l);
}

layer mktest_connected(int argc, char **argv) {
	layer l = {0};
	
      if( argc < 3 )
		printf("	make_connected_layer(batch, inputs, output, activation, batch_normalize, net->adam)\n\n");

 	return(l);
}

layer mktest_crop(int argc, char **argv) {
	layer l = {0};
	
    if( argc < 3 )
		printf("crop_make_crop_layer(batch,h,w,c,crop_height,crop_width,flip, angle, saturation, exposure)\n\n");

 	return(l);
}

layer mktest_cost(int argc, char **argv) {

    int batch = find_int_arg(argc, argv, "-b", 100);
    int inputs = find_int_arg(argc, argv, "-i", 2000);
    int scale = find_float_arg(argc, argv, "-s", 0.5);
    int cost_type = find_int_arg(argc, argv, "-t", 0);
	
    if( argc < 3 )
		printf("\tmake_cost_layer(batch, inputs, type[0..5] , scale)\n"
                "SSE, MASKED, L1, SEG, SMOOTH,WGAN\n\n");

    cost_layer l = make_cost_layer( batch,  inputs, cost_type, scale);

 	return(l);
}

layer mktest_region(int argc, char **argv) {
	layer l = {0};
	
      if( argc < 3 )
		printf("	make_region_layer(batch, w, h, num, classes, coords)\n\n");

 	return(l);
}

layer mktest_yolo(int argc, char **argv) {
	layer l = {0};
	
      if( argc < 3 )
		printf("	make_yolo_layer(batch, w, h, num, total, mask, classes)\n\n");

 	return(l);
}

layer mktest_iseg(int argc, char **argv) {
	layer l = {0};
	
      if( argc < 3 )
		printf("	make_iseg_layer(batch, w, h, classes, ids)\n\n");

 	return(l);
}

layer mktest_detection(int argc, char **argv) {
	layer l = {0};
	
    if( argc < 3 )
		printf("\tmake_detection_layer(batch, inputs, num, side, classes, coords, rescore)\n\n");

 	return(l);
}

layer mktest_softmax(int argc, char **argv) {

    int batch = find_int_arg(argc, argv, "-b", 100);
    int inputs = find_int_arg(argc, argv, "-i", 2000);
    int groups = find_int_arg(argc, argv, "-g", 20);
	
    if( argc < 3 )
		printf("	make_softmax_layer(batch, inputs, groups)\n\n");

    softmax_layer l = make_softmax_layer(batch, inputs, groups);

 	return(l);
}

layer mktest_normalization(int argc, char **argv) {
	layer l = {0};
      if( argc < 3 )
		printf("	make_normalization_layer(batch, w, h, c, size, alpha, beta, kappa)\n\n");

 	return(l);
}

layer mktest_batchnorm(int argc, char **argv) {
	layer l = {0};
      if( argc < 3 )
		printf("	make_batchnorm_layer(batch, w, h, c)\n\n");

 	return(l);
}

layer mktest_maxpool(int argc, char **argv) {
	layer l = {0};
	
    if( argc < 3 )
		printf("\tmaxpool_layer layer = make_maxpool_layer(batch,h,w,c,size,stride,padding)\n\n");

 	return(l);
}

layer mktest_reorg(int argc, char **argv) {
	layer l = {0};
	
      if( argc < 3 )
		printf("\tmake_reorg_layer(batch,w,h,c,stride,reverse, flatten, extra)\n\n");

 	return(l);
}

layer mktest_avgpool(int argc, char **argv) {

    int batch = find_int_arg(argc, argv, "-b", 100);
    int w = find_int_arg(argc, argv, "-w", 200);
    int h = find_int_arg(argc, argv, "-h", 200);
    int c = find_int_arg(argc, argv, "-c", 200);
	
    if( argc < 3 )
		printf("\tavgpool_layer layer = make_avgpool_layer(batch,w,h,c)\n\n");

    avgpool_layer l = make_avgpool_layer(batch,w,h,c);

 	return(l);
}

layer mktest_route(int argc, char **argv) {
	layer l = {0};

    if( argc < 3 )
		printf("\troute_make_route_layer(batch, n, layers, sizes)\n\n");
 	return(l);
}

layer mktest_upsample(int argc, char **argv) {
	
    int batch = find_int_arg(argc, argv, "-b", 100);
    int w = find_int_arg(argc, argv, "-w", 200);
    int h = find_int_arg(argc, argv, "-h", 200);
    int c = find_int_arg(argc, argv, "-c", 200);
    int s = find_int_arg(argc, argv, "-s", 200);
	
    if( argc < 3 )
	    printf("	make_upsample_layer(batch, w, h, c, stride)\n\n");
    layer l = make_upsample_layer(batch,w,h,c,s);

 	return(l);
}

layer mktest_shortcut(int argc, char **argv) {
	layer l = {0};
    if( argc < 3 )
	    printf("	make_shortcut_layer(batch, index, w, h, c, from.out_w, from.out_h, from.out_c)\n\n");

 	return(l);
}

layer mktest_dropout(int argc, char **argv) {

    int batch = find_int_arg(argc, argv, "-b", 100);
    int inputs = find_int_arg(argc, argv, "-i", 2000);
    float probability = find_float_arg(argc, argv, "-p", 0.5);
	
    if( argc < 3 )
	printf("\tmake_dropout_layer(batch, inputs, probability)\n\n");
    dropout_layer l = make_dropout_layer(batch,inputs, probability);

    return(l);
}



layer build_network_cfg(int argc, char **argv)
{
    layer l;
    if ( argc == 2 ) 
        fprintf(stderr, "layer     filters    size              input                output\n");
    do { 
       LAYER_TYPE lt = string_to_layer_type(argv[1]);

       switch(lt) {

        case CONVOLUTIONAL:
            l = mktest_convolutional(argc,argv);
			break;

        case DECONVOLUTIONAL:
            l = mktest_deconvolutional(argc,argv);
			break;

        case LOCAL:
            l = mktest_local(argc,argv);
			break;

        case ACTIVE:
            l = mktest_activation(argc,argv);
			break;

        case LOGXENT:
            l = mktest_logistic(argc,argv);
			break;

        case L2NORM:
            l = mktest_l2norm(argc,argv);
			break;

        case RNN:
            l = mktest_rnn(argc,argv);
			break;

        case GRU:
            l = mktest_gru(argc,argv);
			break;

        case LSTM:
            l = mktest_lstm(argc,argv);
			break;

        case CRNN:
            l = mktest_crnn(argc,argv);
			break;

        case CONNECTED:
            l = mktest_connected(argc,argv);
			break;

        case CROP:
            l = mktest_crop(argc,argv);
			break;

        case COST:
            l = mktest_cost(argc,argv);
			break;

        case REGION:
            l = mktest_region(argc,argv);
			break;

        case YOLO:
            l = mktest_yolo(argc,argv);
			break;

        case ISEG:
            l = mktest_iseg(argc,argv);
			break;

        case DETECTION:
            l = mktest_detection(argc,argv);
			break;

        case SOFTMAX:
            l = mktest_softmax(argc,argv);
			break;

        case NORMALIZATION:
            l = mktest_normalization(argc,argv);
			break;

        case BATCHNORM:
            l = mktest_batchnorm(argc,argv);
			break;

        case MAXPOOL:
            l = mktest_maxpool(argc,argv);
			break;

        case REORG:
            l = mktest_reorg(argc,argv);
			break;

        case AVGPOOL:
            l = mktest_avgpool(argc,argv);
			break;

        case ROUTE:
            l = mktest_route(argc,argv);
			break;

        case UPSAMPLE:
            l = mktest_upsample(argc,argv);
			break;

        case SHORTCUT:
            l = mktest_shortcut(argc,argv);
			break;

        case DROPOUT:
            l = mktest_dropout(argc,argv);
            break;

        default:
            printf("Unknown layer\n");
			break;

       }
    } while(0);
    
    return l;
}

int main(int argc, char **argv) {

    layer *xl;
    int i;
     
    if(argc < 2){
        fprintf(stderr, "usage: %s [layer_type_name] -i [gpuIndex] \n", argv[0]);
        return 0;
    }
    int gpu_index = find_int_arg(argc, argv, "-i", 0);
    if(find_arg(argc, argv, "-nogpu")) {
        gpu_index = -1;
    }

    int nl = find_int_arg(argc, argv, "-n", 1);
    xl = Calloc(nl, layer);
    for( i=0 ; i<nl; i++)
        xl[i] = build_network_cfg(argc,argv);

    sleep(60);    
    exit(0);    
}

