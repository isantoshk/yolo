#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <pthread.h>

#define classes 80
#define N 3
#define in_w 608
#define in_h 608
#define out_w 19
#define out_w 19

struct detection;
typedef struct detection detection;

struct data;
typedef struct data data;

typedef struct box {
    float x, y, w, h;
} box;

// box.h
typedef struct boxabs {
    float left, right, top, bot;
} boxabs;

// box.h
typedef struct dxrep {
    float dt, db, dl, dr;
} dxrep;

// box.h
typedef struct ious {
    float iou, giou, diou, ciou;
    dxrep dx_iou;
    dxrep dx_giou;
} ious;


// box.h
typedef struct detection{
    box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
    float *uc; // Gaussian_YOLOv3 - tx,ty,tw,th uncertainty
    int points; // bit-0 - center, bit-1 - top-left-corner, bit-2 - bottom-right-corner
    float *embeddings;  // embeddings for tracking
    int embedding_size;
    float sim;
    int track_id;    
} detection;

// network.c -batch inference
typedef struct det_num_pair {
    int num;
    detection *dets;
} det_num_pair, *pdet_num_pair;

// matrix.h
typedef struct matrix {
    int rows, cols;
    float **vals;
} matrix;

// data.h
typedef struct data {
    int w, h;
    matrix X;
    matrix y;
    int shallow;
    int *num_boxes;
    box **boxes;
} data;

typedef struct box_label {
    int id;
    int track_id;
    float x, y, w, h;
    float left, right, top, bottom;
} box_label;





void do_nms_sort(detection *dets, int total, int classes, float thresh);