#include <box.h>

static int entry_index(int location, int entry)
{
    int n = location / (out_w*out_h);
    int loc = location % (out_w*out_h);
    return n*w*h*(4+classes+1) + entry*out_w*out_h + loc;
}

int yolo_num_detections(float* tensors, float thresh)
{
	int i, n;
    int count = 0;
    for(n = 0; n < N; ++n){
        for (i = 0; i < out_w*out_h; ++i) {
            int obj_index  = entry_index(n*out_w*out_h + i, 4);
            if(tensors[obj_index] > thresh){
                ++count;
            }
        }
    }
    return count;
}

int num_detections(float* tensors, int w, int h, float thresh)
{
	int i;
    int s = 0;
    for (i = 0; i < N; ++i) {
        s += yolo_num_detections(float* tensors, thresh);
    }
    return s;
}

detection *make_network_boxes(float* tensors, int w, int h, float thresh, int *num)
{

    int nboxes = num_detections(net, int w, int h, thresh);
    if (num) *num = nboxes;
    detection* dets = (detection*)xcalloc(nboxes, sizeof(detection));
    for (i = 0; i < nboxes; ++i) {
        dets[i].prob = (float*)xcalloc(l.classes, sizeof(float));
        dets[i].uc = NULL;
        dets[i].mask = NULL;
		dets[i].embeddings = NULL;
        dets[i].embedding_size = l.embedding_size;
    }
    return dets;
}

void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative, int letter)
{
    int i;
    // network height (or width)
    int new_w = 0;
    // network height (or width)
    int new_h = 0;
    // Compute scale given image w,h vs network w,h
    // I think this "rotates" the image to match network to input image w/h ratio
    // new_h and new_w are really just network width and height
    if (letter) {
        if (((float)netw / w) < ((float)neth / h)) {
            new_w = netw;
            new_h = (h * netw) / w;
        }
        else {
            new_h = neth;
            new_w = (w * neth) / h;
        }
    }
    else {
        new_w = netw;
        new_h = neth;
    }
    // difference between network width and "rotated" width
    float deltaw = netw - new_w;
    // difference between network height and "rotated" height
    float deltah = neth - new_h;
    // ratio between rotated network width and network width
    float ratiow = (float)new_w / netw;
    // ratio between rotated network width and network width
    float ratioh = (float)new_h / neth;
    for (i = 0; i < n; ++i) {

        box b = dets[i].bbox;
        // x = ( x - (deltaw/2)/netw ) / ratiow;
        //   x - [(1/2 the difference of the network width and rotated width) / (network width)]
        b.x = (b.x - deltaw / 2. / netw) / ratiow;
        b.y = (b.y - deltah / 2. / neth) / ratioh;
        // scale to match rotation of incoming image
        b.w *= 1 / ratiow;
        b.h *= 1 / ratioh;

        // relative seems to always be == 1, I don't think we hit this condition, ever.
        if (!relative) {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }

        dets[i].bbox = b;
    }
}

box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride, int new_coords)
{
    box b;
    // ln - natural logarithm (base = e)
    // x` = t.x * lw - i;   // x = ln(x`/(1-x`))   // x - output of previous conv-layer
    // y` = t.y * lh - i;   // y = ln(y`/(1-y`))   // y - output of previous conv-layer
    // w = ln(t.w * net.w / anchors_w); // w - output of previous conv-layer
    // h = ln(t.h * net.h / anchors_h); // h - output of previous conv-layer
    if (new_coords) {
        b.x = (i + x[index + 0 * stride]) / lw;
        b.y = (j + x[index + 1 * stride]) / lh;
        b.w = x[index + 2 * stride] * x[index + 2 * stride] * 4 * biases[2 * n] / w;
        b.h = x[index + 3 * stride] * x[index + 3 * stride] * 4 * biases[2 * n + 1] / h;
    }
    else {
        b.x = (i + x[index + 0 * stride]) / lw;
        b.y = (j + x[index + 1 * stride]) / lh;
        b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
        b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
    }
    return b;
}

int get_yolo_detections(float* tensors, loat thresh, int *map, int relative, detection *dets, int letter)
{
    int i,j,n;
    float *predictions = tensors;
    int count = 0;
    int* mask1[3] = {0, 1, 2}
    int* mask2[3] = {3, 4, 5}
    int* mask3[3] = {6, 7, 8}
    float* biases = {12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401};
    for (i = 0; i < out_w*out_h; ++i){
        int row = i / out_w;
        int col = i % out_w;
        for(n = 0; n < N; ++n){
            int obj_index  = entry_index(l, 0, n*out_w*out_h + i, 4);
            float objectness = predictions[obj_index];
            if (objectness > thresh){
                int box_index = entry_index(l, 0, n*out_w*out_h + i, 0);
                dets[count].bbox = get_yolo_box(predictions, biases, mask3[n], box_index, col, row, out_w, out_h, out_w*out_h, 0);
                dets[count].objectness = objectness;
                dets[count].classes = classes;
                if (l.embedding_output) {
                    get_embedding(l.embedding_output, l.w, l.h, l.n*l.embedding_size, l.embedding_size, col, row, n, 0, dets[count].embeddings);
                }

                for (j = 0; j < l.classes; ++j) {
                    int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                    float prob = objectness*predictions[class_index];
                    dets[count].prob[j] = (prob > thresh) ? prob : 0;
                }
                ++count;
            }
        }
    }
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative, letter);
    return count;
}

void fill_network_boxes(float *tensors, float thresh, float hier, int *map, int relative, detection *dets, int letter)
{
    int j;
    for (j = 0; j < net->n; ++j) {
        layer l = net->layers[j];
        int count = get_yolo_detections(l, thresh, map, relative, dets, letter);
        dets += count;
            
        
    }
}


detection get_network_boxes(float* tensors, float thresh, float hier, int *map, int relative, int *num, int letter)
{
	detection* det = make_network_boxes(tensors, w, h, thresh, num);
	fill_network_boxes(tensors, thresh, hier, map, relative, det, letter);
}

void postprocess(float* tensors, int cells, float thresh, float hier_thresh,  int* map, int relative, int* num, int letter)
{
	int nboxes = 0;
	int nms = 0;
	float beta_nms = 0.6
	//detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num, int letter)
	detection *dets = get_network_boxes(tensors, cells, thresh, hier_thresh, 0, 1, &nboxes, letter_box);
	if (nms) {
        diounms_sort(dets, nboxes, classes, nms, beta_nms);
    }

}