#include "akazed.h"
#include <device_launch_parameters.h>
#include <memory>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>


#define X1 256
#define X2 16
#define NBINS 300
#define MAX_SCALE 5
#define MAX_OCTAVE 8
#define MAX_DIST 96
#define FMIN_VAL (-1E6F)
#define IMIN_VAL -1E6


__device__ unsigned int d_max_contrast;
__device__ int d_hist[NBINS];
__constant__ float d_extrema_param[MAX_SCALE * 2];
__constant__ int d_max_num_points;
__device__ unsigned int d_point_counter;
__constant__ int d_oparams[MAX_OCTAVE * 5];
__constant__ int comp_idx_1[61 * 8];
__constant__ int comp_idx_2[61 * 8];



void getMaxContrastAddr(void** addr)
{
	CHECK(cudaGetSymbolAddress(addr, d_max_contrast));
}


void setHistogram(const int* h_hist)
{
	CHECK(cudaMemcpyToSymbol(d_hist, h_hist, NBINS * sizeof(int), 0, cudaMemcpyHostToDevice));
}


void setExtremaParam(const float* param, const int n)
{
	CHECK(cudaMemcpyToSymbol(d_extrema_param, param, n * sizeof(float), 0, cudaMemcpyHostToDevice));
}


void getPointCounter(void** addr)
{
	CHECK(cudaGetSymbolAddress(addr, d_point_counter));
}


void setMaxNumPoints(const int num)
{
	CHECK(cudaMemcpyToSymbol(d_max_num_points, &num, sizeof(int), 0, cudaMemcpyHostToDevice));
}


void setOparam(const int* oparams, const int n)
{
	CHECK(cudaMemcpyToSymbol(d_oparams, oparams, n * sizeof(int), 0, cudaMemcpyHostToDevice));
}


void setCompareIndices()
{
	int comp_idx_1_h[61 * 8];
	int comp_idx_2_h[61 * 8];

	int cntr = 0, i = 0, j = 0;
	// 2x2
	for (j = 0; j < 4; ++j)
	{
		for (i = j + 1; i < 4; ++i)
		{
			comp_idx_1_h[cntr] = 3 * j;
			comp_idx_2_h[cntr] = 3 * i;
			cntr++;
		}
	}
	for (j = 0; j < 3; ++j)
	{
		for (i = j + 1; i < 4; ++i)
		{
			comp_idx_1_h[cntr] = 3 * j + 1;
			comp_idx_2_h[cntr] = 3 * i + 1;
			cntr++;
		}
	}
	for (j = 0; j < 3; ++j)
	{
		for (i = j + 1; i < 4; ++i)
		{
			comp_idx_1_h[cntr] = 3 * j + 2;
			comp_idx_2_h[cntr] = 3 * i + 2;
			cntr++;
		}
	}

	// 3x3
	for (j = 4; j < 12; ++j)
	{
		for (i = j + 1; i < 13; ++i)
		{
			comp_idx_1_h[cntr] = 3 * j;
			comp_idx_2_h[cntr] = 3 * i;
			cntr++;
		}
	}
	for (j = 4; j < 12; ++j)
	{
		for (i = j + 1; i < 13; ++i)
		{
			comp_idx_1_h[cntr] = 3 * j + 1;
			comp_idx_2_h[cntr] = 3 * i + 1;
			cntr++;
		}
	}
	for (j = 4; j < 12; ++j)
	{
		for (i = j + 1; i < 13; ++i)
		{
			comp_idx_1_h[cntr] = 3 * j + 2;
			comp_idx_2_h[cntr] = 3 * i + 2;
			cntr++;
		}
	}

	// 4x4
	for (j = 13; j < 28; ++j)
	{
		for (i = j + 1; i < 29; ++i)
		{
			comp_idx_1_h[cntr] = 3 * j;
			comp_idx_2_h[cntr] = 3 * i;
			cntr++;
		}
	}
	for (j = 13; j < 28; ++j)
	{
		for (i = j + 1; i < 29; ++i)
		{
			comp_idx_1_h[cntr] = 3 * j + 1;
			comp_idx_2_h[cntr] = 3 * i + 1;
			cntr++;
		}
	}
	for (j = 13; j < 28; ++j)
	{
		for (i = j + 1; i < 29; ++i)
		{
			comp_idx_1_h[cntr] = 3 * j + 2;
			comp_idx_2_h[cntr] = 3 * i + 2;
			cntr++;
		}
	}

	CHECK(cudaMemcpyToSymbol(comp_idx_1, comp_idx_1_h, 8 * 61 * sizeof(int)));
	CHECK(cudaMemcpyToSymbol(comp_idx_2, comp_idx_2_h, 8 * 61 * sizeof(int)));
}


__inline__ __device__ int borderAdd(const int a, const int b, const int m)
{
	const int c = a + b;
	if (c < m)
	{
		return c;
	}
	return m + m - 2 - c;
}


inline __device__ float dFastAtan2(float y, float x)
{
	const float absx = fabs(x);
	const float absy = fabs(y);
	const float a = __fdiv_rn(min(absx, absy), max(absx, absy));
	const float s = a * a;
	float r = __fmaf_rn(__fmaf_rn(__fmaf_rn(-0.0464964749f, s, 0.15931422f), s, -0.327622764f), s * a, a);
	r = (absy > absx ? H_PI - r : r);
	r = (x < 0 ? M_PI - r : r);
	r = (y < 0 ? -r : r);
	return r;
}



namespace akaze
{
	__constant__ float d_lowpass_kernel[21];


	void setLowPassKernel(const float* kernel, const int ksz)
	{
		CHECK(cudaMemcpyToSymbol(d_lowpass_kernel, kernel, ksz * sizeof(float), 0, cudaMemcpyHostToDevice));
	}


	template <int RADIUS>
	__global__ void gConv2d(float* src, float* dst, int width, int height, int pitch)
	{
		__shared__ float sdata[X2 + 2 * RADIUS][X2];
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix = blockIdx.x * blockDim.x + tix;
		int iy = blockIdx.y * blockDim.y + tiy;
		if (ix >= width || iy >= height)
		{
			return;
		}

		int ystart = iy * pitch;
		int idx = ystart + ix;
		int hsubor = height + height - 2;
		int idx0 = idx, idx1 = idx;
		int toy = RADIUS + tiy;
		int br_border = X2 - 1;

		float wsum = src[idx] * d_lowpass_kernel[0];
		for (int i = 1; i <= RADIUS; i++)
		{
			idx0 = abs(ix - i) + ystart;
			idx1 = borderAdd(ix, i, width) + ystart;
			wsum += d_lowpass_kernel[i] * (src[idx0] + src[idx1]);
		}
		sdata[toy][tix] = wsum;

		int new_toy = toy, new_iy = iy;
		bool at_edge = false;
		if (tiy <= RADIUS && tiy > 0)
		{
			at_edge = true;
			new_toy = RADIUS - tiy;
			new_iy = abs(iy - (tiy + tiy));
		}
		else if (toy >= br_border && tiy < br_border)
		{
			at_edge = true;
			new_toy = 2 * (br_border + RADIUS) - toy;
			new_iy = borderAdd(iy, 2 * (br_border - tiy), height);
		}
		else if (iy + RADIUS >= height)
		{
			at_edge = true;
			new_toy = toy + RADIUS;
			new_iy = hsubor - (RADIUS + iy);
		}

		if (at_edge)
		{
			int new_ystart = new_iy * pitch;
			int new_idx = new_ystart + ix;
			wsum = src[new_idx] * d_lowpass_kernel[0];
			for (int i = 1; i <= RADIUS; i++)
			{
				idx0 = abs(ix - i) + new_ystart;
				idx1 = borderAdd(ix, i, width) + new_ystart;
				wsum += d_lowpass_kernel[i] * (src[idx0] + src[idx1]);
			}
			sdata[new_toy][tix] = wsum;
		}
		__syncthreads();

		wsum = sdata[toy][tix] * d_lowpass_kernel[0];
		for (int i = 1; i <= RADIUS; i++)
		{
			wsum += d_lowpass_kernel[i] * (sdata[toy - i][tix] + sdata[toy + i][tix]);
		}
		dst[idx] = wsum;
	}


	__global__ void gDownWithSmooth(float* src, float* dst, float* smooth, int3 swhp, int3 dwhp)
	{
		__shared__ float sdata[X2 + 4][X2];
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int dix = blockIdx.x * blockDim.x + tix;
		int diy = blockIdx.y * blockDim.y + tiy;
		bool in_bounds = (dix < dwhp.x && diy < dwhp.y);
		int six = dix + dix;
		int siy = diy + diy;
		int ystart = siy * swhp.z;
		int toy = tiy + 2;

		int sxes[5] = { abs(six - 4), abs(six - 2), six, borderAdd(six, 2, swhp.x), borderAdd(six, 4, swhp.x) };
		if (in_bounds)
		{
			sdata[toy][tix] = d_lowpass_kernel[0] * src[ystart + sxes[2]] +
				d_lowpass_kernel[1] * (src[ystart + sxes[1]] + src[ystart + sxes[3]]) +
				d_lowpass_kernel[2] * (src[ystart + sxes[0]] + src[ystart + sxes[4]]);
		}
		else
		{
			sdata[toy][tix] = 0.0f;
		}
		__syncthreads();

		int yborder = X2 - 1;
		int new_toy = toy, new_siy = siy;
		bool at_edge = false;
		if (tiy <= 2 && tiy > 0)
		{
			at_edge = true;
			new_toy = 2 - tiy;
			new_siy = abs(siy - 4 * tiy);
		}
		else if (toy >= yborder && tiy < yborder)
		{
			at_edge = true;
			new_toy = 2 * (yborder + 2) - toy;
			new_siy = borderAdd(siy, 4 * (yborder - tiy), swhp.y);
		}
		else if (siy + 4 >= swhp.y)
		{
			at_edge = true;
			new_toy = toy + 2;
			new_siy = swhp.y + swhp.y - 2 - (4 + siy);
		}

		if (at_edge)
		{
			int new_ystart = new_siy * swhp.z;
			if (in_bounds)
			{
				sdata[new_toy][tix] = d_lowpass_kernel[0] * src[new_ystart + sxes[2]] +
					d_lowpass_kernel[1] * (src[new_ystart + sxes[1]] + src[new_ystart + sxes[3]]) +
					d_lowpass_kernel[2] * (src[new_ystart + sxes[0]] + src[new_ystart + sxes[4]]);
			}
			else
			{
				sdata[new_toy][tix] = 0.0f;
			}
		}
		__syncthreads();

		if (in_bounds)
		{
			int didx = diy * dwhp.z + dix;
			dst[didx] = src[ystart + six];
			smooth[didx] = d_lowpass_kernel[0] * sdata[toy][tix] +
				d_lowpass_kernel[1] * (sdata[toy - 1][tix] + sdata[toy + 1][tix]) +
				d_lowpass_kernel[2] * (sdata[toy - 2][tix] + sdata[toy + 2][tix]);
		}
	}


	__global__ void gScharrContrastNaive(float* src, float* dst, int width, int height, int pitch)
	{
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix1 = blockIdx.x * X2 + tix;
		int iy1 = blockIdx.y * X2 + tiy;
		if (ix1 >= width || iy1 >= height)
		{
			return;
		}

		int ix0 = abs(ix1 - 1);
		int ix2 = borderAdd(ix1, 1, width);
		int iy0 = abs(iy1 - 1);
		int iy2 = borderAdd(iy1, 1, height);

		int irow0 = iy0 * pitch;
		int irow1 = iy1 * pitch;
		int irow2 = iy2 * pitch;

		float dx = 10 * (src[irow1 + ix2] - src[irow1 + ix0]) + 3 * (src[irow0 + ix2] + src[irow2 + ix2] - src[irow0 + ix0] - src[irow2 + ix0]);
		float dy = 10 * (src[irow2 + ix1] - src[irow0 + ix1]) + 3 * (src[irow2 + ix0] + src[irow2 + ix2] - src[irow0 + ix0] - src[irow0 + ix2]);
		dst[irow1 + ix1] = __fsqrt_rn(dx * dx + dy * dy);
	}


	__inline__ __device__ void sort2vals(float* src, int i, int j)
	{
		if (src[i] < src[j])
		{
			float temp = src[i];
			src[i] = src[j];
			src[j] = temp;
		}
	}


	__global__ void gFindMaxContrastU4(float* src, int width, int height, int pitch)
	{
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int tid = tiy * X2 + tix;
		int ix0 = blockIdx.x * X2 * 2 + tix;
		int iy0 = blockIdx.y * X2 * 2 + tiy;
		int ix1 = ix0 + X2;
		int iy1 = iy0 + X2;
		bool in_bounds = (ix0 < width && iy0 < height);

		int x0y0 = iy0 * pitch + ix0;
		if (in_bounds && iy1 < height)
		{
			int x0y1 = iy1 * pitch + ix0;
			sort2vals(src, x0y0, x0y1);
			if (ix1 < width)
			{
				int x1y1 = iy1 * pitch + ix1;
				sort2vals(src, x0y0, x1y1);
			}
		}
		if (in_bounds && ix1 < width)
		{
			int x1y0 = iy0 * pitch + ix1;
			sort2vals(src, x0y0, x1y0);
		}

		int block_ox = blockIdx.x * X2 * 2;
		int block_oy = blockIdx.y * X2 * 2;
		for (int stride = X2 * X2 / 2; stride > 0; stride >>= 1)
		{
			if (tid < stride)
			{
				int nid = tid + stride;
				int niy = nid / X2;
				int nix = nid % X2;
				bool nid_in_bounds = (block_ox + nix < width) && (block_oy + niy < height);
				int nidx = niy * pitch + nix;
				if (in_bounds && nid_in_bounds)
				{
					sort2vals(src, x0y0, nidx);
				}
			}
			__syncthreads();
		}
		if (tid == 0 && in_bounds)
		{
			unsigned int* gradi = (unsigned int*)&src[x0y0];
			atomicMax(&d_max_contrast, *gradi);
		}
	}


	__global__ void gConstrastHistShared(float* grad, float factor, int width, int height, int pitch)
	{
		__shared__ int shist[NBINS];

		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix = blockIdx.x * 32 + tix;
		int iy = blockIdx.y * 16 + tiy;
		bool in_bounds = (ix < width && iy < height);

		int tid = tiy * 32 + tix;
		if (tid < NBINS)
		{
			shist[tid] = 0;
		}
		__syncthreads();

		if (in_bounds)
		{
			int idx = iy * pitch + ix;
			int hi = __fmul_rz(grad[idx], factor);
			if (hi >= NBINS)
			{
				hi = NBINS - 1;
			}
			atomicAdd(shist + hi, 1);
		}
		__syncthreads();

		if (tid < NBINS)
		{
			atomicAdd(d_hist + tid, shist[tid]);
		}
	}


	__global__ void gFlowNaive(float* src, float* dst, DiffusivityType type, float ikc, int width, int height, int pitch)
	{
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix1 = blockIdx.x * X2 + tix;
		int iy1 = blockIdx.y * X2 + tiy;
		if (ix1 >= width || iy1 >= height)
		{
			return;
		}

		int ix0 = abs(ix1 - 1);
		int ix2 = borderAdd(ix1, 1, width);
		int iy0 = abs(iy1 - 1);
		int iy2 = borderAdd(iy1, 1, height);

		int irow0 = iy0 * pitch;
		int irow1 = iy1 * pitch;
		int irow2 = iy2 * pitch;

		float dx = 10 * (src[irow1 + ix2] - src[irow1 + ix0]) + 3 * (src[irow0 + ix2] + src[irow2 + ix2] - src[irow0 + ix0] - src[irow2 + ix0]);
		float dy = 10 * (src[irow2 + ix1] - src[irow0 + ix1]) + 3 * (src[irow2 + ix0] + src[irow2 + ix2] - src[irow0 + ix0] - src[irow0 + ix2]);
		float dif2 = ikc * (dx * dx + dy * dy);
		if (type == PM_G1)
		{
			dst[irow1 + ix1] = __expf(-dif2);
		}
		else if (type == PM_G2)
		{
			dst[irow1 + ix1] = 1.f / (1.f + dif2);
		}
		else if (type == WEICKERT)
		{
			dst[irow1 + ix1] = 1.f - __expf(-3.315f / __powf(dif2, 4));
		}
		else
		{
			dst[irow1 + ix1] = 1.f / __fsqrt_rn(1.f + dif2);
		}
	}


	__global__ void gNldStepNaive(float* src, float* flow, float* dst, float stepfac, int width, int height, int pitch)
	{
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix1 = blockIdx.x * X2 + tix;
		int iy1 = blockIdx.y * X2 + tiy;
		if (ix1 >= width || iy1 >= height)
		{
			return;
		}
		int ix0 = abs(ix1 - 1);
		int iy0 = abs(iy1 - 1);
		int ix2 = borderAdd(ix1, 1, width);
		int iy2 = borderAdd(iy1, 1, height);
		int ystart0 = iy0 * pitch;
		int ystart1 = iy1 * pitch;
		int ystart2 = iy2 * pitch;
		int idx1 = ystart1 + ix1;
		float step = (flow[idx1] + flow[ystart1 + ix2]) * (src[ystart1 + ix2] - src[idx1]) +
			(flow[idx1] + flow[ystart1 + ix0]) * (src[ystart1 + ix0] - src[idx1]) +
			(flow[idx1] + flow[ystart2 + ix1]) * (src[ystart2 + ix1] - src[idx1]) +
			(flow[idx1] + flow[ystart0 + ix1]) * (src[ystart0 + ix1] - src[idx1]);
		dst[idx1] = __fmaf_rn(stepfac, step, src[idx1]);
	}


	__global__ void gDerivate(float* src, float* dx, float* dy, int step, float fac1, float fac2, int width, int height, int pitch)
	{
		int ix1 = blockIdx.x * X2 + threadIdx.x;
		int iy1 = blockIdx.y * X2 + threadIdx.y;
		if (ix1 >= width || iy1 >= height)
		{
			return;
		}
		int ix0 = abs(ix1 - step);
		int ix2 = borderAdd(ix1, step, width);
		int iy0 = abs(iy1 - step);
		int iy2 = borderAdd(iy1, step, height);
		int ystart0 = iy0 * pitch;
		int ystart1 = iy1 * pitch;
		int ystart2 = iy2 * pitch;
		int idx = ystart1 + ix1;

		float ul = src[ystart0 + ix0];
		float uc = src[ystart0 + ix1];
		float ur = src[ystart0 + ix2];
		float cl = src[ystart1 + ix0];
		float cr = src[ystart1 + ix2];
		float ll = src[ystart2 + ix0];
		float lc = src[ystart2 + ix1];
		float lr = src[ystart2 + ix2];

		dx[idx] = fac1 * (ur + lr - ul - ll) + fac2 * (cr - cl);
		dy[idx] = fac1 * (lr + ll - ur - ul) + fac2 * (lc - uc);
	}


	__global__ void gHessianDeterminant(float* dx, float* dy, float* detd, int step, float fac1, float fac2, int width, int height, int pitch)
	{
		int ix1 = blockIdx.x * X2 + threadIdx.x;
		int iy1 = blockIdx.y * X2 + threadIdx.y;
		if (ix1 >= width || iy1 >= height)
		{
			return;
		}
		int ix0 = abs(ix1 - step);
		int ix2 = borderAdd(ix1, step, width);
		int iy0 = abs(iy1 - step);
		int iy2 = borderAdd(iy1, step, height);
		int ystart0 = iy0 * pitch;
		int ystart1 = iy1 * pitch;
		int ystart2 = iy2 * pitch;
		int idx = ystart1 + ix1;

		int iul = ystart0 + ix0;
		int iuc = ystart0 + ix1;
		int iur = ystart0 + ix2;
		int icl = ystart1 + ix0;
		int icr = ystart1 + ix2;
		int ill = ystart2 + ix0;
		int ilc = ystart2 + ix1;
		int ilr = ystart2 + ix2;

		float dxx = fac1 * (dx[iur] + dx[ilr] - dx[iul] - dx[ill]) + fac2 * (dx[icr] - dx[icl]);
		float dxy = fac1 * (dx[ilr] + dx[ill] - dx[iur] - dx[iul]) + fac2 * (dx[ilc] - dx[iuc]);
		float dyy = fac1 * (dy[ilr] + dy[ill] - dy[iur] - dy[iul]) + fac2 * (dy[ilc] - dy[iuc]);

		detd[idx] = dxx * dyy - dxy * dxy;
	}


	__global__ void gCalcExtremaMap(float* dets, float* response_map, float* size_map, int* layer_map, int octave, int max_scale,
		int psz, float threshold, int width, int height, int pitch, int opitch)
	{
		int curr_scale = blockIdx.z;
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix = blockIdx.x * X2 + tix + psz;
		int iy = blockIdx.y * X2 + tiy + psz;
		float border = d_extrema_param[curr_scale];
		float size = d_extrema_param[max_scale + curr_scale];

		int left_x = (int)(ix - border + 0.5f) - 1;
		int right_x = (int)(ix + border + 0.5f) + 1;
		int up_y = (int)(iy - border + 0.5f) - 1;
		int down_y = (int)(iy + border + 0.5f) + 1;
		if (left_x < 0 || right_x >= width || up_y < 0 || down_y >= height)
		{
			return;
		}

		float* curr_det = dets + curr_scale * height * pitch;
		int idx = iy * pitch + ix;
		float* vp = curr_det + idx;
		float* vp0 = vp - pitch;
		float* vp2 = vp + pitch;
		if (*vp > threshold && *vp > *vp0 && *vp > *vp2 && *vp > *(vp - 1) && *vp > *(vp + 1) &&
			*vp > *(vp0 - 1) && *vp > *(vp0 + 1) && *vp > *(vp2 - 1) && *vp > *(vp2 + 1))
		{
			int oix = (ix << octave);
			int oiy = (iy << octave);
			int oidx = oiy * opitch + oix;
			unsigned int* addr = (unsigned int*)&response_map[oidx];
			unsigned int old_uint = *addr;
			while (*vp > __uint_as_float(old_uint))
			{
				unsigned int assumed = old_uint;
				old_uint = atomicCAS(addr, assumed, __float_as_uint(*vp));
				if (assumed == old_uint)
				{
					size_map[oidx] = size;
					layer_map[oidx] = octave * max_scale + curr_scale;
					break;
				}
			}
		}
	}


	__global__ void gNmsRNaive(AkazePoint* points, float* response_map, float* size_map, int* layer_map, int psz, int r, int width, int height, int pitch)
	{
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix = blockIdx.x * X2 + tix + psz;
		int iy = blockIdx.y * X2 + tiy + psz;
		if (ix + psz >= width || iy + psz >= height)
		{
			return;
		}

		int ystart = iy * pitch;
		int idx = ystart + ix;
		if (layer_map[idx] >= 0)
		{
			float fsz = size_map[idx];
			int isz = (int)(fsz + 0.5f);
			int sqsz = fsz * fsz;
			int ii = 0, new_idx = 0;
			int new_systart = (iy - isz) * pitch;
			bool to_nms = false;
			for (int i = -isz; i <= isz; i++)
			{
				ii = i * i;
				new_idx = new_systart + ix - isz;
				for (int j = -isz; j <= isz; j++)
				{
					if (i == 0 && j == 0)
					{
						continue;
					}
					if (ii + j * j < sqsz &&
						(response_map[new_idx] > FMIN_VAL &&
							(response_map[new_idx] > response_map[idx] ||
								(response_map[new_idx] == response_map[idx] && i <= 0 && j <= 0)))
						)
					{
						to_nms = true;
					}
					new_idx++;
				}
				if (to_nms)
				{
					break;
				}
				new_systart += pitch;
			}
			if (!to_nms)
			{
				unsigned int pi = atomicInc(&d_point_counter, 0x7fffffff);
				if (pi < d_max_num_points)
				{
					points[pi].x = ix;
					points[pi].y = iy;
					points[pi].octave = layer_map[idx];
					points[pi].size = size_map[idx];
				}
			}
		}
	}


	__global__ void gRefine(AkazePoint* points, float* tmem, int noctaves, int max_scale)
	{
		unsigned int pi = blockIdx.x * X1 + threadIdx.x;
		if (pi >= d_point_counter)
		{
			return;
		}

		int* osizes = d_oparams;
		int* offsets = osizes + noctaves;
		int3* owhps = (int3*)(offsets + noctaves + 1);

		AkazePoint* pt = points + pi;
		int o = pt->octave / max_scale;
		int s = pt->octave % max_scale;
		int p = owhps[o].z;
		float* det = tmem + offsets[o] + (max_scale + s) * osizes[o];
		int y = (int)pt->y >> o;
		int x = (int)pt->x >> o;
		int idx = y * p + x;
		float v2 = det[idx] + det[idx];
		float dx = 0.5f * (det[idx + 1] - det[idx - 1]);
		float dy = 0.5f * (det[idx + p] - det[idx - p]);
		float dxx = det[idx + 1] + det[idx - 1] - v2;
		float dyy = det[idx + p] + det[idx - p] - v2;
		float dxy = 0.25f * (det[idx + p + 1] + det[idx - p - 1] - det[idx - p + 1] - det[idx + p - 1]);
		float dd = dxx * dyy - dxy * dxy;
		float idd = dd != 0.f ? 1.f / dd : 0.f;
		float dst0 = idd * (dxy * dy - dyy * dx);
		float dst1 = idd * (dxy * dx - dxx * dy);
		if (dst0 < -1.f || dst0 > 1.f || dst1 < -1.f || dst1 > 1.f)
		{
			return;
		}
		int ratio = 1 << o;
		pt->y = ratio * (y + dst1);
		pt->x = ratio * (x + dst0);
	}


	__global__ void gCalcOrient(AkazePoint* points, float* tmem, int noctaves, int max_scale)
	{
		__shared__ float resx[42], resy[42];
		__shared__ float re8x[42], re8y[42];
		__shared__ float s_dx[208], s_dy[208];
		__shared__ int s_a[208];
		int pi = blockIdx.x;
		int tix = threadIdx.x;

		if (tix < 42)
		{
			resx[tix] = 0.f;
			resy[tix] = 0.f;
		}
		s_dx[tix] = 0.f;
		s_dy[tix] = 0.f;
		s_a[tix] = -1;
		__syncthreads();

		int* osizes = d_oparams;
		int* offsets = osizes + noctaves;
		int3* owhps = (int3*)(offsets + noctaves + 1);

		AkazePoint* pt = points + pi;
		int o = pt->octave / max_scale;
		int s = pt->octave % max_scale;
		int p = owhps[o].z;
		float* dxd = tmem + offsets[o] + (max_scale * 2 + s) * osizes[o];
		float* dyd = dxd + max_scale * osizes[o];
		int step = (int)(pt->size + 0.5f);
		int x = (int)(pt->x + 0.5f) >> o;
		int y = (int)(pt->y + 0.5f) >> o;
		int i = (tix & 15) - 6;
		int j = (tix / 16) - 6;
		int r2 = i * i + j * j;
		if (r2 < 36)
		{
			float gweight = exp(-r2 * 0.08f);
			int pos = (y + step * j) * p + (x + step * i);
			float dx = gweight * dxd[pos];
			float dy = gweight * dyd[pos];
			float angle = atan2(dy, dx);
			int a = max(min((int)(angle * (21 / M_PI)) + 21, 41), 0);
			s_dx[tix] = dx;
			s_dy[tix] = dy;
			s_a[tix] = a;
		}
		__syncthreads();

		if (tix < 42)
		{
			for (int k = 0; k < 208; k++)
			{
				if (s_a[k] == tix)
				{
					resx[tix] += s_dx[k];
					resy[tix] += s_dy[k];
				}
			}
		}
		__syncthreads();

		if (tix < 42)
		{
			re8x[tix] = resx[tix];
			re8y[tix] = resy[tix];
			for (int k = tix + 1; k < tix + 7; k++)
			{
				re8x[tix] += resx[k < 42 ? k : k - 42];
				re8y[tix] += resy[k < 42 ? k : k - 42];
			}
		}
		__syncthreads();

		if (tix == 0)
		{
			float maxr = 0.0f;
			int maxk = 0;
			for (int k = 0; k < 42; k++)
			{
				float r = re8x[k] * re8x[k] + re8y[k] * re8y[k];
				if (r > maxr)
				{
					maxr = r;
					maxk = k;
				}
			}
			float angle = dFastAtan2(re8y[maxk], re8x[maxk]);
			pt->angle = (angle < 0.0f ? angle + 2.0f * M_PI : angle);
		}
	}


	__global__ void gDescribe2(AkazePoint* points, float* tmem, int noctaves, int max_scale, int size2, int size3, int size4)
	{
#define EXTRACT_S 64
		__shared__ float acc_vals[3 * 30 * EXTRACT_S];
		int pi = blockIdx.x;
		int tix = threadIdx.x;

		float* acc_vals_im = &acc_vals[0];
		float* acc_vals_dx = &acc_vals[30 * EXTRACT_S];
		float* acc_vals_dy = &acc_vals[2 * 30 * EXTRACT_S];

		AkazePoint* pt = points + pi;
		int o = pt->octave / max_scale;
		int s = pt->octave % max_scale;
		float iratio = 1.f / (1 << o);
		int scale = (int)(pt->size + 0.5f);
		float xf = pt->x * iratio;
		float yf = pt->y * iratio;
		float ang = pt->angle;
		float co = __cosf(ang);
		float si = __sinf(ang);

		int* osizes = d_oparams;
		int* offsets = osizes + noctaves;
		int3* owhps = (int3*)(offsets + noctaves + 1);
		int p = owhps[o].z;

		float* imd = tmem + offsets[o] + s * osizes[o];
		float* dxd = imd + max_scale * osizes[o] * 2;
		float* dyd = dxd + max_scale * osizes[o];
		int winsize = max(3 * size3, 4 * size4);

		for (int i = 0; i < 30; ++i)
		{
			int j = i * EXTRACT_S + tix;
			acc_vals_im[j] = 0.f;
			acc_vals_dx[j] = 0.f;
			acc_vals_dy[j] = 0.f;
		}
		__syncthreads();

		for (int i = tix; i < winsize * winsize; i += EXTRACT_S)
		{
			int y = i / winsize;
			int x = i - winsize * y;
			int m = max(x, y);
			if (m >= winsize)
				continue;
			int l = x - size2;
			int k = y - size2;
			int xp = (int)(xf + scale * (k * co - l * si) + 0.5f);
			int yp = (int)(yf + scale * (k * si + l * co) + 0.5f);
			int pos = yp * p + xp;
			float im = imd[pos];
			float dx = dxd[pos];
			float dy = dyd[pos];
			float rx = -dx * si + dy * co;
			float ry = dx * co + dy * si;

			if (m < 2 * size2)
			{
				int x2 = (x < size2 ? 0 : 1);
				int y2 = (y < size2 ? 0 : 1);
				acc_vals[3 * (y2 * 2 + x2) + 3 * 30 * tix] += im;
				acc_vals[3 * (y2 * 2 + x2) + 3 * 30 * tix + 1] += rx;
				acc_vals[3 * (y2 * 2 + x2) + 3 * 30 * tix + 2] += ry;
			}
			if (m < 3 * size3)
			{
				int x3 = (x < size3 ? 0 : (x < 2 * size3 ? 1 : 2));
				int y3 = (y < size3 ? 0 : (y < 2 * size3 ? 1 : 2));
				acc_vals[3 * (4 + y3 * 3 + x3) + 3 * 30 * tix] += im;
				acc_vals[3 * (4 + y3 * 3 + x3) + 3 * 30 * tix + 1] += rx;
				acc_vals[3 * (4 + y3 * 3 + x3) + 3 * 30 * tix + 2] += ry;
			}
			if (m < 4 * size4)
			{
				int x4 = (x < 2 * size4 ? (x < size4 ? 0 : 1) : (x < 3 * size4 ? 2 : 3));
				int y4 = (y < 2 * size4 ? (y < size4 ? 0 : 1) : (y < 3 * size4 ? 2 : 3));
				acc_vals[3 * (4 + 9 + y4 * 4 + x4) + 3 * 30 * tix] += im;
				acc_vals[3 * (4 + 9 + y4 * 4 + x4) + 3 * 30 * tix + 1] += rx;
				acc_vals[3 * (4 + 9 + y4 * 4 + x4) + 3 * 30 * tix + 2] += ry;
			}
		}
		__syncthreads();

		float acc_reg;
#pragma unroll
		for (int i = 0; i < 15; ++i)
		{
			int offset = 2 * i + (tix < 32 ? 0 : 1);
			int tix_d = tix < 32 ? tix : tix - 32;
			for (int d = 0; d < 90; d += 30)
			{
				if (tix_d < 32)
				{
					acc_reg = acc_vals[3 * 30 * tix_d + offset + d] +
						acc_vals[3 * 30 * (tix_d + 32) + offset + d];
					acc_reg += shiftDown(acc_reg, 1);
					acc_reg += shiftDown(acc_reg, 2);
					acc_reg += shiftDown(acc_reg, 4);
					acc_reg += shiftDown(acc_reg, 8);
					acc_reg += shiftDown(acc_reg, 16);
				}
				if (tix_d == 0)
				{
					acc_vals[offset + d] = acc_reg;
				}
			}
		}

		__syncthreads();

		if (tix < 61)
		{
			unsigned char desc_r = 0;
#pragma unroll
			for (int i = 0; i < (tix == 60 ? 6 : 8); ++i)
			{
				int idx1 = comp_idx_1[tix * 8 + i];
				int idx2 = comp_idx_2[tix * 8 + i];
				desc_r |= (acc_vals[idx1] > acc_vals[idx2] ? 1 : 0) << i;
			}
			pt->features[tix] = desc_r;
		}
	}


	inline __device__ int dHammingDistance2(unsigned char* f1, unsigned char* f2)
	{
		int dist = 0;
		unsigned long long* v1 = (unsigned long long*)f1;
		unsigned long long* v2 = (unsigned long long*)f2;
		for (int i = 0; i < 8; i++)
		{
			dist += __popcll(v1[i] ^ v2[i]);
		}
		return dist;
	}


	__global__ void gHammingMatch(AkazePoint* points1, AkazePoint* points2, int n1, int n2)
	{
		__shared__ unsigned char ofeat[(FLEN + 7) & ~7];
		__shared__ int score_1st[X2];
		__shared__ int score_2nd[X2];
		__shared__ int idx_1st[X2];

		unsigned int tid = threadIdx.x;
		unsigned int bid = blockIdx.x;
		if (bid >= n1)
		{
			return;
		}

		AkazePoint* p1 = &points1[bid];

		if (tid == 0)
		{
			for (int i = 0; i < FLEN; i++)
			{
				ofeat[i] = p1->features[i];
			}
			for (int i = FLEN; i < ((FLEN + 7) & ~7); i++)
			{
				ofeat[i] = 0;
			}
		}
		score_1st[tid] = 512;
		score_2nd[tid] = 512;
		idx_1st[tid] = -1;
		__syncthreads();

		int max_iter = (n2 + X2 - 1) / X2;
		for (int k = 0; k < max_iter; k++)
		{
			int pi = k * X2 + tid;
			if (pi < n2)
			{
				AkazePoint* p2 = &points2[pi];
				int dist = dHammingDistance2(ofeat, p2->features);
				if (dist < score_1st[tid])
				{
					score_2nd[tid] = score_1st[tid];
					score_1st[tid] = dist;
					idx_1st[tid] = pi;
				}
				else if (dist < score_2nd[tid])
				{
					score_2nd[tid] = dist;
				}
			}
			__syncthreads();
		}

		for (int stride = X2 / 2; stride > 0; stride >>= 1)
		{
			if (tid < stride)
			{
				int ntid = tid + stride;
				if (score_1st[ntid] < score_1st[tid])
				{
					score_2nd[tid] = score_1st[tid];
					score_1st[tid] = score_1st[ntid];
					idx_1st[tid] = idx_1st[ntid];
				}
				else if (score_1st[ntid] < score_2nd[tid])
				{
					score_2nd[tid] = score_1st[ntid];
				}
				if (score_2nd[ntid] < score_2nd[tid])
				{
					score_2nd[tid] = score_2nd[ntid];
				}
			}
			__syncthreads();
		}

		if (tid == 0)
		{
			if (score_1st[0] < MAX_DIST)
			{
				AkazePoint* p2 = &points2[idx_1st[0]];
				p1->match = idx_1st[0];
				p1->distance = score_1st[0];
				p1->distance2 = score_2nd[0];
				p1->match_x = p2->x;
				p1->match_y = p2->y;
			}
			else
			{
				p1->match = -1;
				p1->distance = -1;
				p1->distance2 = -1;
				p1->match_x = -1;
				p1->match_y = -1;
			}
		}
		__syncthreads();
	}


	void createGaussKernel(float var, int radius)
	{
		static float _var = -1.f;
		static int _radius = 0;
		if (abs(_var - var) < 1e-3 && _radius == radius)
		{
			return;
		}

		_var = var;
		_radius = radius;

		const int ksz = radius + 1;
		std::unique_ptr<float[]> kptr(new float[ksz]);
		float denom = 1.f / (2.f * var);
		float* kernel = kptr.get();
		float ksum = 0;
		for (int i = 0; i < ksz; i++)
		{
			kernel[i] = expf(-i * i * denom);
			if (i == 0)
			{
				ksum += kernel[i];
			}
			else
			{
				ksum += kernel[i] + kernel[i];
			}
		}
		ksum = 1 / ksum;
		for (int i = 0; i < ksz; i++)
		{
			kernel[i] *= ksum;
		}
		setLowPassKernel(kernel, ksz);
	}


	void hLowPass(float* src, float* dst, int width, int height, int pitch, float var, int ksz)
	{
		dim3 block(X2, X2);
		dim3 grid((width + X2 - 1) / X2, (height + X2 - 1) / X2);
		if (ksz <= 5)
		{
			createGaussKernel(var, 2);
			gConv2d<2> << <grid, block >> > (src, dst, width, height, pitch);
		}
		else if (ksz <= 7)
		{
			createGaussKernel(var, 3);
			gConv2d<3> << <grid, block >> > (src, dst, width, height, pitch);
		}
		else if (ksz <= 9)
		{
			createGaussKernel(var, 4);
			gConv2d<4> << <grid, block >> > (src, dst, width, height, pitch);
		}
		else if (ksz <= 11)
		{
			createGaussKernel(var, 5);
			gConv2d<5> << <grid, block >> > (src, dst, width, height, pitch);
		}
		else
		{
			std::cerr << "Kernels larger than 11 not implemented" << std::endl;
		}
	}


	void hDownWithSmooth(float* src, float* dst, float* smooth, int3 swhp, int3 dwhp)
	{
		createGaussKernel(1.f, 2);
		dim3 block(X2, X2);
		dim3 grid((dwhp.x + X2 - 1) / X2, (dwhp.y + X2 - 1) / X2);
		gDownWithSmooth << <grid, block >> > (src, dst, smooth, swhp, dwhp);
		CHECK(cudaDeviceSynchronize());
		CheckMsg("hScaleDown() execution failed\n");
	}


	void hScharrContrast(float* src, float* grad, float& kcontrast, float per, int width, int height, int pitch)
	{
		float h_max_contrast = 0.03f;
		unsigned int* d_max_contrast_addr;
		getMaxContrastAddr((void**)&d_max_contrast_addr);
		CHECK(cudaMemcpy(d_max_contrast_addr, &h_max_contrast, sizeof(float), cudaMemcpyHostToDevice));

		dim3 block(X2, X2);
		dim3 grid((width + X2 - 1) / X2, (height + X2 - 1) / X2);
		gScharrContrastNaive << <grid, block >> > (src, grad, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

		dim3 grid1((width / 2 + X2 - 1) / X2, (height / 2 + X2 - 1) / X2);
		gFindMaxContrastU4 << <grid1, block >> > (grad, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

		CHECK(cudaMemcpy(&h_max_contrast, d_max_contrast_addr, sizeof(float), cudaMemcpyDeviceToHost));

		int h_hist[NBINS];
		memset(h_hist, 0, NBINS * sizeof(int));
		CHECK(cudaMemcpyToSymbol(d_hist, h_hist, NBINS * sizeof(int), 0, cudaMemcpyHostToDevice));

		float hfactor = NBINS / h_max_contrast;
		dim3 block2(32, 16);
		dim3 grid2((width + 32 - 1) / 32, (height + 16 - 1) / 16);
		gConstrastHistShared << <grid2, block2 >> > (grad, hfactor, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

		CHECK(cudaMemcpyFromSymbol(h_hist, d_hist, NBINS * sizeof(int), 0, cudaMemcpyDeviceToHost));

		int thresh = (width * height - h_hist[0]) * per;
		int cumuv = 0;
		int k = 1;
		while (k < NBINS)
		{
			if (cumuv >= thresh)
			{
				break;
			}
			cumuv += h_hist[k];
			k++;
		}
		kcontrast = k / hfactor;

		CheckMsg("hScharrContrast() execution failed\n");
	}


	void hFlow(float* src, float* flow, DiffusivityType type, float kcontrast, int width, int height, int pitch)
	{
		float ikc = 1.f / (kcontrast * kcontrast);
		dim3 block(X2, X2);
		dim3 grid((width + X2 - 1) / X2, (height + X2 - 1) / X2);
		gFlowNaive << <grid, block >> > (src, flow, type, ikc, width, height, pitch);
		CHECK(cudaDeviceSynchronize());
		CheckMsg("hFlow() execution failed\n");
	}


	void hNldStep(float* img, float* flow, float* temp, float step_size, int width, int height, int pitch)
	{
		float stepfac = 0.5f * step_size;
		dim3 block(X2, X2);
		dim3 grid((width + X2 - 1) / X2, (height + X2 - 1) / X2);
		gNldStepNaive << <grid, block >> > (img, flow, temp, stepfac, width, height, pitch);
		CHECK(cudaDeviceSynchronize());
		CheckMsg("hNldStep() execution failed\n");
	}


	void hHessianDeterminant(float* src, float* dx, float* dy, int step, int width, int height, int pitch)
	{
		float w = 10.f / 3.f;
		float fac1 = 1.f / (2.f * (w + 2.f));
		float fac2 = w * fac1;

		dim3 block(X2, X2);
		dim3 grid((width + X2 - 1) / X2, (height + X2 - 1) / X2);
		gDerivate << <grid, block >> > (src, dx, dy, step, fac1, fac2, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

		gHessianDeterminant<<<grid, block>>>(dx, dy, src, step, fac1, fac2, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

		CheckMsg("hHessianDeterminant() execution failed\n");
	}


	void hCalcExtremaMap(float* dets, float* response_map, float* size_map, int* layer_map, float* params,
		int octave, int max_scale, float threshold, int width, int height, int pitch, int opitch)
	{
		setExtremaParam(params, max_scale * 2);

		int psz = (int)params[0];
		int depad = psz * 2;

		dim3 block(X2, X2);
		dim3 grid((width - depad + X2 - 1) / X2, (height - depad + X2 - 1) / X2, max_scale);
		gCalcExtremaMap << <grid, block >> > (dets, response_map, size_map, layer_map, octave, max_scale, psz,
			threshold, width, height, pitch, opitch);
		CHECK(cudaDeviceSynchronize());
		CheckMsg("hCalcExtremaMap() execution failed\n");
	}


	void hNmsR(AkazePoint* points, float* response_map, float* size_map, int* layer_map, int psz, int neigh, int width, int height, int pitch)
	{
		int psz2 = psz + psz;
		dim3 block(X2, X2);
		dim3 grid((width - psz2 + X2 - 1) / X2, (height - psz2 + X2 - 1) / X2);
		int shared_radius = X2 + 2 * neigh;
		size_t shared_nbytes = shared_radius * shared_radius * sizeof(float);
		gNmsRNaive << <grid, block, shared_nbytes >> > (points, response_map, size_map, layer_map, psz, neigh, width, height, pitch);
		CHECK(cudaDeviceSynchronize());
		CheckMsg("hNmsR() execution failed\n");
	}


	struct AkazePointCompare
	{
		__host__ __device__ bool operator()(const AkazePoint& a, const AkazePoint& b) const
		{
			if (a.y != b.y) return a.y < b.y;
			return a.x < b.x;
		}
	};

	void sortAkazePoints(AkazePoint* d_points, int num_pts)
	{
		if (num_pts <= 0) return;
		thrust::device_ptr<AkazePoint> ptr(d_points);
		thrust::sort(thrust::device, ptr, ptr + num_pts, AkazePointCompare());
	}


	void hRefine(AkazeData& result, float* tmem, int noctaves, int max_scale)
	{
		dim3 block(X1);
		dim3 grid((result.num_pts + X1 - 1) / X1);
		gRefine << <grid, block >> > (result.d_data, tmem, noctaves, max_scale);
		CHECK(cudaDeviceSynchronize());
		CheckMsg("hRefine() execution failed\n");
	}


	void hCalcOrient(AkazeData& result, float* tmem, int noctaves, int max_scale)
	{
		dim3 block(13 * 16);
		dim3 grid(result.num_pts);
		gCalcOrient << <grid, block >> > (result.d_data, tmem, noctaves, max_scale);
		CHECK(cudaDeviceSynchronize());
		CheckMsg("hCalcOrient() execution failed\n");
	}


	void hDescribe(AkazeData& result, float* tmem, int noctaves, int max_scale, int patsize)
	{
		int size2 = patsize;
		int size3 = ceilf(2.0f * patsize / 3.0f);
		int size4 = ceilf(0.5f * patsize);

		dim3 block(64);
		dim3 grid(result.num_pts);
		gDescribe2 << <grid, block >> > (result.d_data, tmem, noctaves, max_scale, size2, size3, size4);
		CHECK(cudaDeviceSynchronize());
		CheckMsg("hDescribe() execution failed\n");
	}


	void hMatch(AkazeData& result1, AkazeData& result2)
	{
		dim3 block(X2);
		dim3 grid(result1.num_pts);
		gHammingMatch << <grid, block >> > (result1.d_data, result2.d_data, result1.num_pts, result2.num_pts);
		CHECK(cudaDeviceSynchronize());
		CheckMsg("hMatch() execution failed\n");
	}

}


namespace fastakaze
{
	__constant__ int d_lowpass_kernel[21];


	__global__ void gConv2dR2(unsigned char* src, int* dst, int width, int height, int pitch)
	{
#define MX (X2 - 1)
		__shared__ int sdata[X2 + 2 * 2][X2];
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix = blockIdx.x * blockDim.x + tix;
		int iy = blockIdx.y * blockDim.y + tiy;
		if (ix >= width || iy >= height)
		{
			return;
		}

		int k0 = d_lowpass_kernel[0];
		int k1 = d_lowpass_kernel[1];
		int k2 = d_lowpass_kernel[2];

		int ystart = iy * pitch;
		int ixl2 = abs(ix - 2);
		int ixl1 = abs(ix - 1);
		int ixr1 = borderAdd(ix, 1, width);
		int ixr2 = borderAdd(ix, 2, width);

		int toy = tiy + 2;
		sdata[toy][tix] = (k0 * (int)src[ystart + ix] +
			k1 * ((int)src[ystart + ixl1] + (int)src[ystart + ixr1]) +
			k2 * ((int)src[ystart + ixl2] + (int)src[ystart + ixr2])) >> 16;

		int new_toy = toy, new_iy = iy;
		bool at_edge = false;
		if (tiy <= 2 && tiy > 0)
		{
			at_edge = true;
			new_toy = 2 - tiy;
			new_iy = abs(iy - (tiy + tiy));
		}
		else if (toy >= MX && tiy < MX)
		{
			at_edge = true;
			new_toy = 2 * (MX + 2) - toy;
			new_iy = borderAdd(iy, 2 * (MX - tiy), height);
		}
		else if (iy + 2 >= height)
		{
			at_edge = true;
			new_toy = toy + 2;
			new_iy = height + height - 2 - (2 + iy);
		}

		if (at_edge)
		{
			int new_ystart = new_iy * pitch;
			sdata[new_toy][tix] = (k0 * (int)src[new_ystart + ix] +
				k1 * ((int)src[new_ystart + ixl1] + (int)src[new_ystart + ixr1]) +
				k2 * ((int)src[new_ystart + ixl2] + (int)src[new_ystart + ixr2])) >> 16;
		}
		__syncthreads();

		dst[ystart + ix] = (k0 * sdata[toy][tix] +
			k1 * (sdata[toy - 1][tix] + sdata[toy + 1][tix]) +
			k2 * (sdata[toy - 2][tix] + sdata[toy + 2][tix])) >> 16;
	}


	__global__ void gConv2dR2(int* src, int* dst, int width, int height, int pitch)
	{
		__shared__ int sdata[X2 + 2 * 2][X2];
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix = blockIdx.x * blockDim.x + tix;
		int iy = blockIdx.y * blockDim.y + tiy;
		if (ix >= width || iy >= height)
		{
			return;
		}

		int k0 = d_lowpass_kernel[0];
		int k1 = d_lowpass_kernel[1];
		int k2 = d_lowpass_kernel[2];

		int ystart = iy * pitch;
		int ixl2 = abs(ix - 2);
		int ixl1 = abs(ix - 1);
		int ixr1 = borderAdd(ix, 1, width);
		int ixr2 = borderAdd(ix, 2, width);

		int toy = tiy + 2;
		sdata[toy][tix] = (k0 * src[ystart + ix] +
			k1 * (src[ystart + ixl1] + src[ystart + ixr1]) +
			k2 * (src[ystart + ixl2] + src[ystart + ixr2])) >> 16;

		int new_toy = toy, new_iy = iy;
		bool at_edge = false;
		if (tiy <= 2 && tiy > 0)
		{
			at_edge = true;
			new_toy = 2 - tiy;
			new_iy = abs(iy - (tiy + tiy));
		}
		else if (toy >= MX && tiy < MX)
		{
			at_edge = true;
			new_toy = 2 * (MX + 2) - toy;
			new_iy = borderAdd(iy, 2 * (MX - tiy), height);
		}
		else if (iy + 2 >= height)
		{
			at_edge = true;
			new_toy = toy + 2;
			new_iy = height + height - 2 - (2 + iy);
		}

		if (at_edge)
		{
			int new_ystart = new_iy * pitch;
			sdata[new_toy][tix] = (k0 * src[new_ystart + ix] +
				k1 * (src[new_ystart + ixl1] + src[new_ystart + ixr1]) +
				k2 * (src[new_ystart + ixl2] + src[new_ystart + ixr2])) >> 16;
		}
		__syncthreads();

		dst[ystart + ix] = (k0 * sdata[toy][tix] +
			k1 * (sdata[toy - 1][tix] + sdata[toy + 1][tix]) +
			k2 * (sdata[toy - 2][tix] + sdata[toy + 2][tix])) >> 16;
	}


	template <int RADIUS>
	__global__ void gConv2d(unsigned char* src, int* dst, int width, int height, int pitch)
	{
		__shared__ int sdata[X2 + 2 * RADIUS][X2];
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix = blockIdx.x * blockDim.x + tix;
		int iy = blockIdx.y * blockDim.y + tiy;
		if (ix >= width || iy >= height)
		{
			return;
		}

		int ystart = iy * pitch;
		int idx = ystart + ix;
		int hsubor = height + height - 2;
		int idx0 = idx, idx1 = idx;
		int toy = RADIUS + tiy;
		int br_border = X2 - 1;

		int wsum = d_lowpass_kernel[0] * (int)src[idx];
		for (int i = 1; i <= RADIUS; i++)
		{
			idx0 = abs(ix - i) + ystart;
			idx1 = borderAdd(ix, i, width) + ystart;
			wsum += d_lowpass_kernel[i] * ((int)src[idx0] + (int)src[idx1]);
		}
		sdata[toy][tix] = wsum >> 16;

		int new_toy = toy, new_iy = iy;
		bool at_edge = false;
		if (tiy <= RADIUS && tiy > 0)
		{
			at_edge = true;
			new_toy = RADIUS - tiy;
			new_iy = abs(iy - (tiy + tiy));
		}
		else if (toy >= br_border && tiy < br_border)
		{
			at_edge = true;
			new_toy = 2 * (br_border + RADIUS) - toy;
			new_iy = borderAdd(iy, 2 * (br_border - tiy), height);
		}
		else if (iy + RADIUS >= height)
		{
			at_edge = true;
			new_toy = toy + RADIUS;
			new_iy = hsubor - (RADIUS + iy);
		}

		if (at_edge)
		{
			int new_ystart = new_iy * pitch;
			int new_idx = new_ystart + ix;
			wsum = d_lowpass_kernel[0] * (int)src[new_idx];
			for (int i = 1; i <= RADIUS; i++)
			{
				idx0 = abs(ix - i) + new_ystart;
				idx1 = borderAdd(ix, i, width) + new_ystart;
				wsum += d_lowpass_kernel[i] * ((int)src[idx0] + (int)src[idx1]);
			}
			sdata[new_toy][tix] = wsum >> 16;
		}
		__syncthreads();

		wsum = d_lowpass_kernel[0] * sdata[toy][tix];
		for (int i = 1; i <= RADIUS; i++)
		{
			wsum += d_lowpass_kernel[i] * (sdata[toy - i][tix] + sdata[toy + i][tix]);
		}
		dst[idx] = wsum >> 16;
	}


	__global__ void gDownWithSmooth(int* src, int* dst, int* smooth, int3 swhp, int3 dwhp)
	{
		__shared__ int sdata[X2 + 4][X2];
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int dix = blockIdx.x * blockDim.x + tix;
		int diy = blockIdx.y * blockDim.y + tiy;
		bool in_bounds = (dix < dwhp.x && diy < dwhp.y);
		int six = dix + dix;
		int siy = diy + diy;
		int ystart = siy * swhp.z;
		int toy = tiy + 2;

		int sxes[5] = { abs(six - 4), abs(six - 2), six, borderAdd(six, 2, swhp.x), borderAdd(six, 4, swhp.x) };
		if (in_bounds)
		{
			sdata[toy][tix] = (d_lowpass_kernel[0] * src[ystart + sxes[2]] +
				d_lowpass_kernel[1] * (src[ystart + sxes[1]] + src[ystart + sxes[3]]) +
				d_lowpass_kernel[2] * (src[ystart + sxes[0]] + src[ystart + sxes[4]])) >> 16;
		}
		else
		{
			sdata[toy][tix] = 0;
		}
		__syncthreads();

		int yborder = X2 - 1;
		int new_toy = toy, new_siy = siy;
		bool at_edge = false;
		if (tiy <= 2 && tiy > 0)
		{
			at_edge = true;
			new_toy = 2 - tiy;
			new_siy = abs(siy - 4 * tiy);
		}
		else if (toy >= yborder && tiy < yborder)
		{
			at_edge = true;
			new_toy = 2 * (yborder + 2) - toy;
			new_siy = borderAdd(siy, 4 * (yborder - tiy), swhp.y);
		}
		else if (siy + 4 >= swhp.y)
		{
			at_edge = true;
			new_toy = toy + 2;
			new_siy = swhp.y + swhp.y - 2 - (4 + siy);
		}

		if (at_edge)
		{
			int new_ystart = new_siy * swhp.z;
			if (in_bounds)
			{
				sdata[new_toy][tix] = (d_lowpass_kernel[0] * src[new_ystart + sxes[2]] +
					d_lowpass_kernel[1] * (src[new_ystart + sxes[1]] + src[new_ystart + sxes[3]]) +
					d_lowpass_kernel[2] * (src[new_ystart + sxes[0]] + src[new_ystart + sxes[4]])) >> 16;
			}
			else
			{
				sdata[new_toy][tix] = 0;
			}
		}
		__syncthreads();

		if (in_bounds)
		{
			int didx = diy * dwhp.z + dix;
			dst[didx] = src[ystart + six];
			smooth[didx] = (d_lowpass_kernel[0] * sdata[toy][tix] +
				d_lowpass_kernel[1] * (sdata[toy - 1][tix] + sdata[toy + 1][tix]) +
				d_lowpass_kernel[2] * (sdata[toy - 2][tix] + sdata[toy + 2][tix])) >> 16;
		}
	}


	__global__ void gScharrContrastNaive(int* src, int* dst, int width, int height, int pitch)
	{
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix1 = blockIdx.x * X2 + tix;
		int iy1 = blockIdx.y * X2 + tiy;
		if (ix1 >= width || iy1 >= height)
		{
			return;
		}

		int ix0 = abs(ix1 - 1);
		int ix2 = borderAdd(ix1, 1, width);
		int iy0 = abs(iy1 - 1);
		int iy2 = borderAdd(iy1, 1, height);

		int irow0 = iy0 * pitch;
		int irow1 = iy1 * pitch;
		int irow2 = iy2 * pitch;

		int dx = 10 * (src[irow1 + ix2] - src[irow1 + ix0]) + 3 * (src[irow0 + ix2] + src[irow2 + ix2] - src[irow0 + ix0] - src[irow2 + ix0]);
		int dy = 10 * (src[irow2 + ix1] - src[irow0 + ix1]) + 3 * (src[irow2 + ix0] + src[irow2 + ix2] - src[irow0 + ix0] - src[irow0 + ix2]);
		dst[irow1 + ix1] = (int)(__fsqrt_rn(dx * dx + dy * dy) + 0.5f);
	}


	__inline__ __device__ void sort2vals(int* src, int i, int j)
	{
		if (src[i] < src[j])
		{
			int temp = src[i];
			src[i] = src[j];
			src[j] = temp;
		}
	}


	__global__ void gFindMaxContrastU4(int* src, int width, int height, int pitch)
	{
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int tid = tiy * X2 + tix;
		int ix0 = blockIdx.x * X2 * 2 + tix;
		int iy0 = blockIdx.y * X2 * 2 + tiy;
		int ix1 = ix0 + X2;
		int iy1 = iy0 + X2;
		bool in_bounds = (ix0 < width && iy0 < height);

		int x0y0 = iy0 * pitch + ix0;
		if (in_bounds && iy1 < height)
		{
			int x0y1 = iy1 * pitch + ix0;
			sort2vals(src, x0y0, x0y1);
			if (ix1 < width)
			{
				int x1y1 = iy1 * pitch + ix1;
				sort2vals(src, x0y0, x1y1);
			}
		}
		if (in_bounds && ix1 < width)
		{
			int x1y0 = iy0 * pitch + ix1;
			sort2vals(src, x0y0, x1y0);
		}

		int block_ox = blockIdx.x * X2 * 2;
		int block_oy = blockIdx.y * X2 * 2;
		for (int stride = X2 * X2 / 2; stride > 0; stride >>= 1)
		{
			if (tid < stride)
			{
				int nid = tid + stride;
				int niy = nid / X2;
				int nix = nid % X2;
				bool nid_in_bounds = (block_ox + nix < width) && (block_oy + niy < height);
				int nidx = niy * pitch + nix;
				if (in_bounds && nid_in_bounds)
				{
					sort2vals(src, x0y0, nidx);
				}
			}
			__syncthreads();
		}
		if (tid == 0 && in_bounds)
		{
			atomicMax(&d_max_contrast, src[x0y0]);
		}
	}


	__global__ void gConstrastHistShared(int* grad, int factor, int width, int height, int pitch)
	{
		__shared__ int shist[NBINS];

		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix = blockIdx.x * 32 + tix;
		int iy = blockIdx.y * 16 + tiy;
		bool in_bounds = (ix < width && iy < height);

		int tid = tiy * 32 + tix;
		if (tid < NBINS)
		{
			shist[tid] = 0;
		}
		__syncthreads();

		if (in_bounds)
		{
			int idx = iy * pitch + ix;
			int hi = (grad[idx] * factor) >> 16;
			if (hi >= NBINS)
			{
				hi = NBINS - 1;
			}
			atomicAdd(shist + hi, 1);
		}
		__syncthreads();

		if (tid < NBINS)
		{
			atomicAdd(d_hist + tid, shist[tid]);
		}
	}


	__global__ void gDerivate(int* src, int* dx, int* dy, int step, int fac1, int fac2, int width, int height, int pitch)
	{
		int ix1 = blockIdx.x * X2 + threadIdx.x;
		int iy1 = blockIdx.y * X2 + threadIdx.y;
		if (ix1 >= width || iy1 >= height)
		{
			return;
		}
		int idx = iy1 * pitch + ix1;
		int ix0 = abs(ix1 - step);
		int ix2 = borderAdd(ix1, step, width);
		int iy0 = abs(iy1 - step);
		int iy2 = borderAdd(iy1, step, height);
		int ystart0 = iy0 * pitch;
		int ystart1 = iy1 * pitch;
		int ystart2 = iy2 * pitch;

		int ul = src[ystart0 + ix0];
		int uc = src[ystart0 + ix1];
		int ur = src[ystart0 + ix2];
		int cl = src[ystart1 + ix0];
		int cr = src[ystart1 + ix2];
		int ll = src[ystart2 + ix0];
		int lc = src[ystart2 + ix1];
		int lr = src[ystart2 + ix2];

		dx[idx] = (fac1 * (ur + lr - ul - ll) + fac2 * (cr - cl)) >> 16;
		dy[idx] = (fac1 * (lr + ll - ur - ul) + fac2 * (lc - uc)) >> 16;
	}


	__global__ void gHessianDeterminant(int* dx, int* dy, int* detd, int step, int fac1, int fac2, int width, int height, int pitch)
	{
		int ix1 = blockIdx.x * X2 + threadIdx.x;
		int iy1 = blockIdx.y * X2 + threadIdx.y;
		if (ix1 >= width || iy1 >= height)
		{
			return;
		}
		int idx = iy1 * pitch + ix1;
		int ix0 = abs(ix1 - step);
		int ix2 = borderAdd(ix1, step, width);
		int iy0 = abs(iy1 - step);
		int iy2 = borderAdd(iy1, step, height);
		int ystart0 = iy0 * pitch;
		int ystart1 = iy1 * pitch;
		int ystart2 = iy2 * pitch;

		int iul = ystart0 + ix0;
		int iuc = ystart0 + ix1;
		int iur = ystart0 + ix2;
		int icl = ystart1 + ix0;
		int icr = ystart1 + ix2;
		int ill = ystart2 + ix0;
		int ilc = ystart2 + ix1;
		int ilr = ystart2 + ix2;

		int dxx = (fac1 * (dx[iur] + dx[ilr] - dx[iul] - dx[ill]) + fac2 * (dx[icr] - dx[icl])) >> 16;
		int dxy = (fac1 * (dx[ilr] + dx[ill] - dx[iur] - dx[iul]) + fac2 * (dx[ilc] - dx[iuc])) >> 16;
		int dyy = (fac1 * (dy[ilr] + dy[ill] - dy[iur] - dy[iul]) + fac2 * (dy[ilc] - dy[iuc])) >> 16;

		detd[idx] = dxx * dyy - dxy * dxy;
	}


	__global__ void gFlowNaive(int* src, int* dst, akaze::DiffusivityType type, float ikc, int width, int height, int pitch)
	{
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix1 = blockIdx.x * X2 + tix;
		int iy1 = blockIdx.y * X2 + tiy;
		if (ix1 >= width || iy1 >= height)
		{
			return;
		}

		int ix0 = abs(ix1 - 1);
		int ix2 = borderAdd(ix1, 1, width);
		int iy0 = abs(iy1 - 1);
		int iy2 = borderAdd(iy1, 1, height);

		int irow0 = iy0 * pitch;
		int irow1 = iy1 * pitch;
		int irow2 = iy2 * pitch;

		int dx = 10 * (src[irow1 + ix2] - src[irow1 + ix0]) + 3 * (src[irow0 + ix2] + src[irow2 + ix2] - src[irow0 + ix0] - src[irow2 + ix0]);
		int dy = 10 * (src[irow2 + ix1] - src[irow0 + ix1]) + 3 * (src[irow2 + ix0] + src[irow2 + ix2] - src[irow0 + ix0] - src[irow0 + ix2]);
		float dif2 = (dx * dx + dy * dy) * ikc;
		if (type == akaze::PM_G1)
		{
			dst[irow1 + ix1] = (int)(__expf(-dif2) * 65536 + 0.5f);
		}
		else if (type == akaze::PM_G2)
		{
			dst[irow1 + ix1] = (int)(1.f / (1.f + dif2) * 65536 + 0.5f);
		}
		else if (type == akaze::WEICKERT)
		{
			dst[irow1 + ix1] = (int)((1.f - __expf(-3.315f / __powf(dif2, 4))) * 65536 + 0.5f);
		}
		else
		{
			dst[irow1 + ix1] = (int)(1.f / __fsqrt_rn(1.f + dif2) * 65536 + 0.5f);
		}
	}


	__global__ void gNldStepNaive(int* src, int* flow, int* dst, int stepfac, int width, int height, int pitch)
	{
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix1 = blockIdx.x * X2 + tix;
		int iy1 = blockIdx.y * X2 + tiy;
		if (ix1 >= width || iy1 >= height)
		{
			return;
		}
		int ix0 = abs(ix1 - 1);
		int iy0 = abs(iy1 - 1);
		int ix2 = borderAdd(ix1, 1, width);
		int iy2 = borderAdd(iy1, 1, height);
		int ystart0 = iy0 * pitch;
		int ystart1 = iy1 * pitch;
		int ystart2 = iy2 * pitch;
		int idx1 = ystart1 + ix1;
		int step = ((flow[idx1] + flow[ystart1 + ix2]) * (src[ystart1 + ix2] - src[idx1]) +
			(flow[idx1] + flow[ystart1 + ix0]) * (src[ystart1 + ix0] - src[idx1]) +
			(flow[idx1] + flow[ystart2 + ix1]) * (src[ystart2 + ix1] - src[idx1]) +
			(flow[idx1] + flow[ystart0 + ix1]) * (src[ystart0 + ix1] - src[idx1])) >> 16;
		dst[idx1] = ((stepfac * step) >> 16) + src[idx1];
	}


	__global__ void gCalcExtremaMap(int* dets, int* response_map, float* size_map, int* layer_map, int octave, int max_scale,
		int psz, int threshold, int width, int height, int pitch, int opitch)
	{
		int curr_scale = blockIdx.z;
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix = blockIdx.x * X2 + tix + psz;
		int iy = blockIdx.y * X2 + tiy + psz;
		float border = d_extrema_param[curr_scale];
		float size = d_extrema_param[max_scale + curr_scale];

		int left_x = (int)(ix - border + 0.5f) - 1;
		int right_x = (int)(ix + border + 0.5f) + 1;
		int up_y = (int)(iy - border + 0.5f) - 1;
		int down_y = (int)(iy + border + 0.5f) + 1;
		if (left_x < 0 || right_x >= width || up_y < 0 || down_y >= height)
		{
			return;
		}

		int* curr_det = dets + curr_scale * height * pitch;
		int idx = iy * pitch + ix;
		int* vp = curr_det + idx;
		int* vp0 = vp - pitch;
		int* vp2 = vp + pitch;
		if (*vp > threshold && *vp > *vp0 && *vp > *vp2 && *vp > *(vp - 1) && *vp > *(vp + 1) &&
			*vp > *(vp0 - 1) && *vp > *(vp0 + 1) && *vp > *(vp2 - 1) && *vp > *(vp2 + 1))
		{
			int oix = (ix << octave);
			int oiy = (iy << octave);
			int oidx = oiy * opitch + oix;
			int old_val = atomicMax(&response_map[oidx], *vp);
			if (old_val < *vp)
			{
				size_map[oidx] = size;
				layer_map[oidx] = octave * max_scale + curr_scale;
			}
		}
	}


	__global__ void gNmsRNaive(akaze::AkazePoint* points, int* response_map, float* size_map, int* layer_map, int psz, int r, int width, int height, int pitch)
	{
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix = blockIdx.x * X2 + tix + psz;
		int iy = blockIdx.y * X2 + tiy + psz;
		if (ix + psz >= width || iy + psz >= height)
		{
			return;
		}

		int ystart = iy * pitch;
		int idx = ystart + ix;
		if (layer_map[idx] >= 0)
		{
			float fsz = size_map[idx];
			int isz = (int)(fsz + 0.5f);
			int sqsz = fsz * fsz;
			int ii = 0, new_idx = 0;
			int new_systart = (iy - isz) * pitch;
			bool to_nms = false;
			for (int i = -isz; i <= isz; i++)
			{
				ii = i * i;
				new_idx = new_systart + ix - isz;
				for (int j = -isz; j <= isz; j++)
				{
					if (i == 0 && j == 0)
					{
						continue;
					}
					if (ii + j * j < sqsz &&
						(response_map[new_idx] > IMIN_VAL &&
							(response_map[new_idx] > response_map[idx] ||
								(response_map[new_idx] == response_map[idx] && i <= 0 && j <= 0)))
						)
					{
						to_nms = true;
					}
					new_idx++;
				}
				if (to_nms)
				{
					break;
				}
				new_systart += pitch;
			}
			if (!to_nms && d_point_counter < d_max_num_points)
			{
				unsigned int pi = atomicInc(&d_point_counter, 0x7fffffff);
				if (pi < d_max_num_points)
				{
					points[pi].x = ix;
					points[pi].y = iy;
					points[pi].octave = layer_map[idx];
					points[pi].size = size_map[idx];
				}
			}
		}
	}


	__global__ void gRefine(akaze::AkazePoint* points, void* tmem, int noctaves, int max_scale)
	{
		unsigned int pi = blockIdx.x * X1 + threadIdx.x;
		if (pi >= d_point_counter)
		{
			return;
		}

		int* osizes = d_oparams;
		int* offsets = osizes + noctaves;
		int3* owhps = (int3*)(offsets + noctaves + 1);

		akaze::AkazePoint* pt = points + pi;
		int o = pt->octave / max_scale;
		int s = pt->octave % max_scale;
		int p = owhps[o].z;
		int* det = (int*)tmem + offsets[o] + (max_scale + s) * osizes[o];
		int y = (int)pt->y >> o;
		int x = (int)pt->x >> o;
		int idx = y * p + x;
		int v2 = det[idx] + det[idx];
		int dx = (det[idx + 1] - det[idx - 1]) >> 1;
		int dy = (det[idx + p] - det[idx - p]) >> 1;
		int dxx = det[idx + 1] + det[idx - 1] - v2;
		int dyy = det[idx + p] + det[idx - p] - v2;
		int dxy = (det[idx + p + 1] + det[idx - p - 1] - det[idx - p + 1] - det[idx + p - 1]) >> 2;
		int dd = dxx * dyy - dxy * dxy;
		float idd = dd != 0 ? (1.f / dd) : 0.f;
		float dst0 = idd * (dxy * dy - dyy * dx);
		float dst1 = idd * (dxy * dx - dxx * dy);
		if (dst0 < -1.f || dst0 > 1.f || dst1 < -1.f || dst1 > 1.f)
		{
			return;
		}
		int ratio = 1 << o;
		pt->y = ratio * (y + dst1);
		pt->x = ratio * (x + dst0);
	}


	__global__ void gCalcOrient(akaze::AkazePoint* points, void* tmem, int noctaves, int max_scale)
	{
		__shared__ float resx[42], resy[42];
		__shared__ float re8x[42], re8y[42];
		__shared__ float s_dx[208], s_dy[208];
		__shared__ int s_a[208];
		int pi = blockIdx.x;
		int tix = threadIdx.x;

		if (tix < 42)
		{
			resx[tix] = 0.f;
			resy[tix] = 0.f;
		}
		s_dx[tix] = 0.f;
		s_dy[tix] = 0.f;
		s_a[tix] = -1;
		__syncthreads();

		int* osizes = d_oparams;
		int* offsets = osizes + noctaves;
		int3* owhps = (int3*)(offsets + noctaves + 1);

		akaze::AkazePoint* pt = points + pi;
		int o = pt->octave / max_scale;
		int s = pt->octave % max_scale;
		int p = owhps[o].z;
		int* dxd = (int*)tmem + offsets[o] + (max_scale * 2 + s) * osizes[o];
		int* dyd = dxd + max_scale * osizes[o];
		int step = (int)(pt->size + 0.5f);
		int x = (int)(pt->x + 0.5f) >> o;
		int y = (int)(pt->y + 0.5f) >> o;
		int i = (tix & 15) - 6;
		int j = (tix / 16) - 6;
		int r2 = i * i + j * j;
		if (r2 < 36)
		{
			float gweight = __expf(-r2 * 0.08f);
			int pos = (y + step * j) * p + (x + step * i);
			float dx = gweight * dxd[pos];
			float dy = gweight * dyd[pos];
			float angle = dFastAtan2(dy, dx);
			int a = max(min((int)(angle * (21 / M_PI)) + 21, 41), 0);
			s_dx[tix] = dx;
			s_dy[tix] = dy;
			s_a[tix] = a;
		}
		__syncthreads();

		if (tix < 42)
		{
			for (int k = 0; k < 208; k++)
			{
				if (s_a[k] == tix)
				{
					resx[tix] += s_dx[k];
					resy[tix] += s_dy[k];
				}
			}
		}
		__syncthreads();

		if (tix < 42)
		{
			re8x[tix] = resx[tix];
			re8y[tix] = resy[tix];
			for (int k = tix + 1; k < tix + 7; k++)
			{
				re8x[tix] += resx[k < 42 ? k : k - 42];
				re8y[tix] += resy[k < 42 ? k : k - 42];
			}
		}
		__syncthreads();

		if (tix == 0)
		{
			float maxr = 0.0f;
			int maxk = 0;
			for (int k = 0; k < 42; k++)
			{
				float r = re8x[k] * re8x[k] + re8y[k] * re8y[k];
				if (r > maxr)
				{
					maxr = r;
					maxk = k;
				}
			}
			float angle = dFastAtan2(re8y[maxk], re8x[maxk]);
			pt->angle = (angle < 0.0f ? angle + 2.0f * M_PI : angle);
		}
	}


	__global__ void gDescribe2(akaze::AkazePoint* points, void* tmem, int noctaves, int max_scale, int size2, int size3, int size4)
	{
		__shared__ int acc_vals[3 * 30 * EXTRACT_S];
		int pi = blockIdx.x;
		int tix = threadIdx.x;

		int* acc_vals_im = &acc_vals[0];
		int* acc_vals_dx = &acc_vals[30 * EXTRACT_S];
		int* acc_vals_dy = &acc_vals[2 * 30 * EXTRACT_S];

		akaze::AkazePoint* pt = points + pi;
		int o = pt->octave / max_scale;
		int s = pt->octave % max_scale;
		float iratio = 1.f / (1 << o);
		int scale = (int)(pt->size + 0.5f);
		float xf = pt->x * iratio;
		float yf = pt->y * iratio;
		float ang = pt->angle;
		float co = __cosf(ang);
		float si = __sinf(ang);

		int* osizes = d_oparams;
		int* offsets = osizes + noctaves;
		int3* owhps = (int3*)(offsets + noctaves + 1);
		int p = owhps[o].z;

		int* imd = (int*)tmem + offsets[o] + s * osizes[o];
		int* dxd = imd + max_scale * osizes[o] * 2;
		int* dyd = dxd + max_scale * osizes[o];
		int winsize = max(3 * size3, 4 * size4);

		for (int i = 0; i < 30; ++i)
		{
			int j = i * EXTRACT_S + tix;
			acc_vals_im[j] = 0;
			acc_vals_dx[j] = 0;
			acc_vals_dy[j] = 0;
		}
		__syncthreads();

		for (int i = tix; i < winsize * winsize; i += EXTRACT_S)
		{
			int y = i / winsize;
			int x = i - winsize * y;
			int m = max(x, y);
			if (m >= winsize)
				continue;
			int l = x - size2;
			int k = y - size2;
			int xp = (int)(xf + scale * (k * co - l * si) + 0.5f);
			int yp = (int)(yf + scale * (k * si + l * co) + 0.5f);
			int pos = yp * p + xp;
			int im = imd[pos];
			int dx = dxd[pos];
			int dy = dyd[pos];
			int rx = -dx * si + dy * co;
			int ry = dx * co + dy * si;

			if (m < 2 * size2)
			{
				int x2 = (x < size2 ? 0 : 1);
				int y2 = (y < size2 ? 0 : 1);
				acc_vals[3 * (y2 * 2 + x2) + 3 * 30 * tix] += im;
				acc_vals[3 * (y2 * 2 + x2) + 3 * 30 * tix + 1] += rx;
				acc_vals[3 * (y2 * 2 + x2) + 3 * 30 * tix + 2] += ry;
			}
			if (m < 3 * size3)
			{
				int x3 = (x < size3 ? 0 : (x < 2 * size3 ? 1 : 2));
				int y3 = (y < size3 ? 0 : (y < 2 * size3 ? 1 : 2));
				acc_vals[3 * (4 + y3 * 3 + x3) + 3 * 30 * tix] += im;
				acc_vals[3 * (4 + y3 * 3 + x3) + 3 * 30 * tix + 1] += rx;
				acc_vals[3 * (4 + y3 * 3 + x3) + 3 * 30 * tix + 2] += ry;
			}
			if (m < 4 * size4)
			{
				int x4 = (x < 2 * size4 ? (x < size4 ? 0 : 1) : (x < 3 * size4 ? 2 : 3));
				int y4 = (y < 2 * size4 ? (y < size4 ? 0 : 1) : (y < 3 * size4 ? 2 : 3));
				acc_vals[3 * (4 + 9 + y4 * 4 + x4) + 3 * 30 * tix] += im;
				acc_vals[3 * (4 + 9 + y4 * 4 + x4) + 3 * 30 * tix + 1] += rx;
				acc_vals[3 * (4 + 9 + y4 * 4 + x4) + 3 * 30 * tix + 2] += ry;
			}
		}
		__syncthreads();

		int acc_reg;
#pragma unroll
		for (int i = 0; i < 15; ++i)
		{
			int offset = 2 * i + (tix < 32 ? 0 : 1);
			int tix_d = tix < 32 ? tix : tix - 32;
			for (int d = 0; d < 90; d += 30)
			{
				if (tix_d < 32)
				{
					acc_reg = acc_vals[3 * 30 * tix_d + offset + d] +
						acc_vals[3 * 30 * (tix_d + 32) + offset + d];
					acc_reg += shiftDown(acc_reg, 1);
					acc_reg += shiftDown(acc_reg, 2);
					acc_reg += shiftDown(acc_reg, 4);
					acc_reg += shiftDown(acc_reg, 8);
					acc_reg += shiftDown(acc_reg, 16);
				}
				if (tix_d == 0)
				{
					acc_vals[offset + d] = acc_reg;
				}
			}
		}

		__syncthreads();

		if (tix < 61)
		{
			unsigned char desc_r = 0;
#pragma unroll
			for (int i = 0; i < (tix == 60 ? 6 : 8); ++i)
			{
				int idx1 = comp_idx_1[tix * 8 + i];
				int idx2 = comp_idx_2[tix * 8 + i];
				desc_r |= (acc_vals[idx1] > acc_vals[idx2] ? 1 : 0) << i;
			}
			pt->features[tix] = desc_r;
		}
	}


	void createGaussKernel(float var, int radius)
	{
		static float _var = -1.f;
		static int _radius = 0;
		if (abs(_var - var) < 1e-3 && _radius == radius)
		{
			return;
		}

		_var = var;
		_radius = radius;

		const int ksz = radius + 1;
		std::unique_ptr<float[]> kptr(new float[ksz]);
		std::unique_ptr<int[]> ikptr(new int[ksz]);
		float denom = 1.f / (2.f * var);
		float* kernel = kptr.get();
		int* ikernel = ikptr.get();
		float ksum = 0;
		for (int i = 0; i < ksz; i++)
		{
			kernel[i] = expf(-i * i * denom);
			if (i == 0)
			{
				ksum += kernel[i];
			}
			else
			{
				ksum += kernel[i] + kernel[i];
			}
		}
		ksum = 1 / ksum;
		for (int i = 0; i < ksz; i++)
		{
			kernel[i] *= ksum;
			ikernel[i] = (int)(kernel[i] * 65536 + 0.5f);
		}

		CHECK(cudaMemcpyToSymbol(d_lowpass_kernel, ikernel, ksz * sizeof(int), 0, cudaMemcpyHostToDevice));
	}


	void hConv2dR2(unsigned char* src, int* dst, int width, int height, int pitch, float var)
	{
		createGaussKernel(var, 2);
		dim3 block(X2, X2);
		dim3 grid((width + X2 - 1) / X2, (height + X2 - 1) / X2);
		gConv2dR2 << <grid, block >> > (src, dst, width, height, pitch);
		CHECK(cudaDeviceSynchronize());
		CheckMsg("hConv2dR2() execution failed\n");
	}


	void hConv2dR2(int* src, int* dst, int width, int height, int pitch, float var)
	{
		createGaussKernel(var, 2);
		dim3 block(X2, X2);
		dim3 grid((width + X2 - 1) / X2, (height + X2 - 1) / X2);
		gConv2dR2 << <grid, block >> > (src, dst, width, height, pitch);
		CHECK(cudaDeviceSynchronize());
		CheckMsg("hConv2dR2() execution failed\n");
	}


	void hLowPass(unsigned char* src, int* dst, int width, int height, int pitch, float var, int ksz)
	{
		dim3 block(X2, X2);
		dim3 grid((width + X2 - 1) / X2, (height + X2 - 1) / X2);
		if (ksz <= 5)
		{
			createGaussKernel(var, 2);
			gConv2d<2> << <grid, block >> > (src, dst, width, height, pitch);
		}
		else if (ksz <= 7)
		{
			createGaussKernel(var, 3);
			gConv2d<3> << <grid, block >> > (src, dst, width, height, pitch);
		}
		else if (ksz <= 9)
		{
			createGaussKernel(var, 4);
			gConv2d<4> << <grid, block >> > (src, dst, width, height, pitch);
		}
		else if (ksz <= 11)
		{
			createGaussKernel(var, 5);
			gConv2d<5> << <grid, block >> > (src, dst, width, height, pitch);
		}
		else
		{
			std::cerr << "Kernels larger than 11 not implemented" << std::endl;
		}
	}


	void hDownWithSmooth(int* src, int* dst, int* smooth, int3 swhp, int3 dwhp)
	{
		createGaussKernel(1.f, 2);
		dim3 block(X2, X2);
		dim3 grid((dwhp.x + X2 - 1) / X2, (dwhp.y + X2 - 1) / X2);
		gDownWithSmooth << <grid, block >> > (src, dst, smooth, swhp, dwhp);
		CHECK(cudaDeviceSynchronize());
		CheckMsg("hDownWithSmooth() execution failed\n");
	}


	void hScharrContrast(int* src, int* grad, int& kcontrast, float per, int width, int height, int pitch)
	{
		int h_max_contrast = 1;
		int* d_max_contrast_addr = NULL;
		CHECK(cudaGetSymbolAddress((void**)&d_max_contrast_addr, d_max_contrast));
		CHECK(cudaMemcpy(d_max_contrast_addr, &h_max_contrast, sizeof(int), cudaMemcpyHostToDevice));

		dim3 block(X2, X2);
		dim3 grid((width + X2 - 1) / X2, (height + X2 - 1) / X2);
		gScharrContrastNaive << <grid, block >> > (src, grad, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

		dim3 grid1((width / 2 + X2 - 1) / X2, (height / 2 + X2 - 1) / X2);
		gFindMaxContrastU4 << <grid1, block >> > (grad, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

		CHECK(cudaMemcpy(&h_max_contrast, d_max_contrast_addr, sizeof(float), cudaMemcpyDeviceToHost));

		int h_hist[NBINS];
		memset(h_hist, 0, NBINS * sizeof(int));
		CHECK(cudaMemcpyToSymbol(d_hist, h_hist, NBINS * sizeof(int), 0, cudaMemcpyHostToDevice));

		int hfactor = (int)(NBINS / (float)h_max_contrast * 65536 + 0.5f);
		dim3 block2(32, 16);
		dim3 grid2((width + 32 - 1) / 32, (height + 16 - 1) / 16);
		gConstrastHistShared << <grid2, block2 >> > (grad, hfactor, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

		CHECK(cudaMemcpyFromSymbol(h_hist, d_hist, NBINS * sizeof(int), 0, cudaMemcpyDeviceToHost));

		int thresh = (width * height - h_hist[0]) * per;
		int cumuv = 0;
		int k = 1;
		while (k < NBINS)
		{
			if (cumuv >= thresh)
			{
				break;
			}
			cumuv += h_hist[k];
			k++;
		}
		kcontrast = k * h_max_contrast / NBINS;

		CheckMsg("hScharrContrast() execution failed\n");
	}


	void hHessianDeterminant(int* src, int* dx, int* dy, int step, int width, int height, int pitch)
	{
		float w = 10.f / 3.f;
		float fac1 = 1.f / (2.f * (w + 2.f));
		float fac2 = w * fac1;
		int ifac1 = (int)(fac1 * 65536 + 0.5f);
		int ifac2 = (int)(fac2 * 65536 + 0.5f);

		dim3 block(X2, X2);
		dim3 grid((width + X2 - 1) / X2, (height + X2 - 1) / X2);
		gDerivate << <grid, block >> > (src, dx, dy, step, ifac1, ifac2, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

		gHessianDeterminant << <grid, block >> > (dx, dy, src, step, ifac1, ifac2, width, height, pitch);
		CHECK(cudaDeviceSynchronize());

		CheckMsg("hHessianDeterminant() execution failed\n");
	}


	void hFlow(int* src, int* flow, akaze::DiffusivityType type, int kcontrast, int width, int height, int pitch)
	{
		float ikc = 1.f / (kcontrast * kcontrast);
		dim3 block(X2, X2);
		dim3 grid((width + X2 - 1) / X2, (height + X2 - 1) / X2);
		gFlowNaive << <grid, block >> > (src, flow, type, ikc, width, height, pitch);
		CHECK(cudaDeviceSynchronize());
		CheckMsg("hFlow() execution failed\n");
	}


	void hNldStep(int* img, int* flow, int* temp, float step_size, int width, int height, int pitch)
	{
		int stepfac = (int)(0.5f * step_size * 65536 + 0.5f);
		dim3 block(X2, X2);
		dim3 grid((width + X2 - 1) / X2, (height + X2 - 1) / X2);
		gNldStepNaive << <grid, block >> > (img, flow, temp, stepfac, width, height, pitch);
		CHECK(cudaDeviceSynchronize());
		CheckMsg("hNldStep() execution failed\n");
	}


	void hCalcExtremaMap(int* dets, int* response_map, float* size_map, int* layer_map, float* params,
		int octave, int max_scale, int threshold, int width, int height, int pitch, int opitch)
	{
		CHECK(cudaMemcpyToSymbol(d_extrema_param, params, max_scale * 2 * sizeof(float), 0, cudaMemcpyHostToDevice));

		int psz = (int)params[0];
		int depad = psz * 2;

		dim3 block(X2, X2);
		dim3 grid((width - depad + X2 - 1) / X2, (height - depad + X2 - 1) / X2, max_scale);
		gCalcExtremaMap << <grid, block >> > (dets, response_map, size_map, layer_map, octave, max_scale, psz,
			threshold, width, height, pitch, opitch);
		CHECK(cudaDeviceSynchronize());
		CheckMsg("hCalcExtremaMap() execution failed\n");
	}


	void hNmsR(akaze::AkazePoint* points, int* response_map, float* size_map, int* layer_map, int psz, int neigh, int width, int height, int pitch)
	{
		int psz2 = psz + psz;
		dim3 block(X2, X2);
		dim3 grid((width - psz2 + X2 - 1) / X2, (height - psz2 + X2 - 1) / X2);
		int shared_radius = X2 + 2 * neigh;
		size_t shared_nbytes = shared_radius * shared_radius * sizeof(float);
		gNmsRNaive << <grid, block, shared_nbytes >> > (points, response_map, size_map, layer_map, psz, neigh, width, height, pitch);
		CHECK(cudaDeviceSynchronize());
		CheckMsg("hNmsR() execution failed\n");
	}


	void hRefine(akaze::AkazeData& result, void* tmem, int noctaves, int max_scale)
	{
		dim3 block(X1);
		dim3 grid((result.num_pts + X1 - 1) / X1);
		gRefine << <grid, block >> > (result.d_data, tmem, noctaves, max_scale);
		CHECK(cudaDeviceSynchronize());
		CheckMsg("hRefine() execution failed\n");
	}


	void hCalcOrient(akaze::AkazeData& result, void* tmem, int noctaves, int max_scale)
	{
		dim3 block(13 * 16);
		dim3 grid(result.num_pts);
		gCalcOrient << <grid, block >> > (result.d_data, tmem, noctaves, max_scale);
		CHECK(cudaDeviceSynchronize());
		CheckMsg("hCalcOrient() execution failed\n");
	}


	void hDescribe(akaze::AkazeData& result, void* tmem, int noctaves, int max_scale, int patsize)
	{
		int size2 = patsize;
		int size3 = ceilf(2.0f * patsize / 3.0f);
		int size4 = ceilf(0.5f * patsize);

		dim3 block(64);
		dim3 grid(result.num_pts);
		gDescribe2 << <grid, block >> > (result.d_data, tmem, noctaves, max_scale, size2, size3, size4);
		CHECK(cudaDeviceSynchronize());
		CheckMsg("hDescribe() execution failed\n");
	}

}
