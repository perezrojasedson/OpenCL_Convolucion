__kernel void conv2d(
    __global const float* input,
    __global float* output,
    __constant float* kdata,
    int width,
    int height,
    int ksize
)
{
    int gx = (int)get_global_id(0);
    int gy = (int)get_global_id(1);

    if (gx >= width || gy >= height) return;

    int khalf = ksize / 2;
    float sum = 0.0f;

    for (int ky = -khalf; ky <= khalf; ky++) {
        for (int kx = -khalf; kx <= khalf; kx++) {
            int ix = gx + kx;
            int iy = gy + ky;

            if (ix < 0) ix = 0;
            if (ix >= width) ix = width - 1;
            if (iy < 0) iy = 0;
            if (iy >= height) iy = height - 1;

            float pixel = input[iy * width + ix];
            float kval = kdata[(ky + khalf) * ksize + (kx + khalf)];
            sum += pixel * kval;
        }
    }

    output[gy * width + gx] = sum;
}
