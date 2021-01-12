#include "box.h"
#include "utils.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.141592
#endif


float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}


float box_iou(box a, box b)
{
    //return box_intersection(a, b)/box_union(a, b);

    float I = box_intersection(a, b);
    float U = box_union(a, b);
    if (I == 0 || U == 0) {
        return 0;
    }
    return I / U;
}

int nms_comparator_v3(const void *pa, const void *pb)
{
    detection a = *(detection *)pa;
    detection b = *(detection *)pb;
    float diff = 0;
    if (b.sort_class >= 0) {
        diff = a.prob[b.sort_class] - b.prob[b.sort_class]; // there is already: prob = objectness*prob
    }
    else {
        diff = a.objectness - b.objectness;
    }
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}

float box_diounms(box a, box b, float beta1)
{
    boxabs ba = box_c(a, b);
    float w = ba.right - ba.left;
    float h = ba.bot - ba.top;
    float c = w * w + h * h;
    float iou = box_iou(a, b);
    if (c == 0) {
        return iou;
    }
    float d = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
    float u = pow(d / c, beta1);
    float diou_term = u;
#ifdef DEBUG_PRINTS
    printf("  c: %f, u: %f, riou_term: %f\n", c, u, diou_term);
#endif
    return iou - diou_term;
}

void diounms_sort(detection *dets, int total, int classes, float thresh, float beta1)
{
    int i, j, k;
    k = total - 1;
    for (i = 0; i <= k; ++i) {
        if (dets[i].objectness == 0) {
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k + 1;

    for (k = 0; k < classes; ++k) {
        for (i = 0; i < total; ++i) {
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(detection), nms_comparator_v3);
        for (i = 0; i < total; ++i)
        {
            if (dets[i].prob[k] == 0) continue;
            box a = dets[i].bbox;
            for (j = i + 1; j < total; ++j) {
                box b = dets[j].bbox;
                // if (box_iou(a, b) > thresh && nms_kind == CORNERS_NMS)
                // {
                //     float sum_prob = pow(dets[i].prob[k], 2) + pow(dets[j].prob[k], 2);
                //     float alpha_prob = pow(dets[i].prob[k], 2) / sum_prob;
                //     float beta_prob = pow(dets[j].prob[k], 2) / sum_prob;
                //     //dets[i].bbox.x = (dets[i].bbox.x*alpha_prob + dets[j].bbox.x*beta_prob);
                //     //dets[i].bbox.y = (dets[i].bbox.y*alpha_prob + dets[j].bbox.y*beta_prob);
                //     //dets[i].bbox.w = (dets[i].bbox.w*alpha_prob + dets[j].bbox.w*beta_prob);
                //     //dets[i].bbox.h = (dets[i].bbox.h*alpha_prob + dets[j].bbox.h*beta_prob);
                    
                //     if (dets[j].points == YOLO_CENTER && (dets[i].points & dets[j].points) == 0) {
                //         dets[i].bbox.x = (dets[i].bbox.x*alpha_prob + dets[j].bbox.x*beta_prob);
                //         dets[i].bbox.y = (dets[i].bbox.y*alpha_prob + dets[j].bbox.y*beta_prob);
                //     }
                //     else if ((dets[i].points & dets[j].points) == 0) {
                //         dets[i].bbox.w = (dets[i].bbox.w*alpha_prob + dets[j].bbox.w*beta_prob);
                //         dets[i].bbox.h = (dets[i].bbox.h*alpha_prob + dets[j].bbox.h*beta_prob);
                //     }
                //     dets[i].points |= dets[j].points;
                    
                //     dets[j].prob[k] = 0;
                // }
                else if (box_diou(a, b) > thresh) {
                    dets[j].prob[k] = 0;
                }
                // else {
                //     if (box_diounms(a, b, beta1) > thresh && nms_kind == DIOU_NMS) {
                //         dets[j].prob[k] = 0;
                //     }
                // }
            }

            //if ((nms_kind == CORNERS_NMS) && (dets[i].points != (YOLO_CENTER | YOLO_LEFT_TOP | YOLO_RIGHT_BOTTOM)))
            //    dets[i].prob[k] = 0;
        }
    }
}
