#include "akaze.h"
#include "akazed.h"
#include "fed.h"
#include <memory>
#include <cmath>
#include <algorithm>
#include <vector>


namespace akaze
{
	void initAkazeData(AkazeData& data, const int max_pts, const bool host, const bool dev)
	{
		data.num_pts = 0;
		data.max_pts = max_pts;
		const size_t size = sizeof(AkazePoint) * max_pts;
		data.h_data = host ? (AkazePoint*)malloc(size) : NULL;
		data.d_data = NULL;
		if (dev)
		{
			CHECK(cudaMalloc((void**)&data.d_data, size));
			CHECK(cudaMemset(data.d_data, 0, size));
		}
	}


	void freeAkazeData(AkazeData& data)
	{
		if (data.d_data != NULL)
		{
			CHECK(cudaFree(data.d_data));
		}
		if (data.h_data != NULL)
		{
			free(data.h_data);
		}
		data.num_pts = 0;
		data.max_pts = 0;
	}


	void cuMatch(AkazeData& result1, AkazeData& result2)
	{
		CHECK(cudaDeviceSynchronize());
		hMatch(result1, result2);
		if (result1.h_data)
		{
			int* h_ptr = &result1.h_data[0].match;
			int* d_ptr = &result1.d_data[0].match;
			CHECK(cudaMemcpy2D(h_ptr, sizeof(AkazePoint), d_ptr, sizeof(AkazePoint), 5 * sizeof(int), result1.num_pts, cudaMemcpyDeviceToHost));
		}
	}


	Akazer::Akazer()
	{
	}


	Akazer::~Akazer()
	{
		CHECK(cudaFree(omem));
	}


	void Akazer::init(int3 whp0, int _noctaves, int _max_scale, float _per, float _kcontrast, float _soffset, bool _reordering,
		float _derivative_factor, float _dthreshold, int _diffusivity, int _descriptor_pattern_size)
	{
		whp.x = whp0.x;
		whp.y = whp0.y;
		whp.z = whp0.z;
		noctaves = _noctaves;
		max_scale = _max_scale;
		per = _per;
		kcontrast = _kcontrast;
		soffset = _soffset;
		reordering = _reordering;
		derivative_factor = _derivative_factor;
		dthreshold = _dthreshold;
		diffusivity = DiffusivityType(_diffusivity);
		descriptor_pattern_size = _descriptor_pattern_size;

		setCompareIndices();
	}


	void Akazer::detectAndCompute(float* image, AkazeData& result, int3 whp0, const bool desc)
	{
		std::vector<int> oparams(noctaves * 5 + 1);
		int* osizes = oparams.data();
		int* offsets = osizes + noctaves;
		int3* owhps = (int3*)(offsets + noctaves + 1);
		float* tmem = NULL;
		const bool reused = whp0.x == whp.x && whp0.y == whp.y;
		if (reused)
		{
			this->allocMemory((void**)&omem, whp0, owhps, osizes, offsets, reused);
			tmem = omem;
		}
		else
		{
			this->allocMemory((void**)&tmem, whp0, owhps, osizes, offsets, reused);
		}

		this->detect(result, tmem, image, owhps, osizes, offsets);

		if (desc)
		{
			hCalcOrient(result, tmem, noctaves, max_scale);
			hDescribe(result, tmem, noctaves, max_scale, descriptor_pattern_size);
		}

		if (result.h_data != NULL)
		{
			float* h_ptr = &result.h_data[0].x;
			float* d_ptr = &result.d_data[0].x;
			CHECK(cudaMemcpy2D(h_ptr, sizeof(AkazePoint), d_ptr, sizeof(AkazePoint), (desc ? FLEN * sizeof(unsigned char) : 0) + 6 * sizeof(float), result.num_pts, cudaMemcpyDeviceToHost));
		}

		if (reused)
		{
			CHECK(cudaMemset(omem, 0, total_osize));
		}
		else
		{
			CHECK(cudaFree(tmem));
		}
	}


	void Akazer::fastDetectAndCompute(unsigned char* image, AkazeData& result, int3 whp0, const bool desc)
	{
		std::vector<int> oparams(noctaves * 5 + 1);
		int* osizes = oparams.data();
		int* offsets = osizes + noctaves;
		int3* owhps = (int3*)(offsets + noctaves + 1);
		void* tmem = NULL;
		const bool reused = whp0.x == whp.x && whp0.y == whp.y;
		if (reused)
		{
			this->allocMemory((void**)&omem, whp0, owhps, osizes, offsets, reused);
			tmem = omem;
		}
		else
		{
			this->allocMemory((void**)&tmem, whp0, owhps, osizes, offsets, reused);
		}

		this->fastDetect(result, tmem, image, owhps, osizes, offsets);

		if (desc)
		{
			fastakaze::hCalcOrient(result, tmem, noctaves, max_scale);
			fastakaze::hDescribe(result, tmem, noctaves, max_scale, descriptor_pattern_size);
		}

		if (result.h_data != NULL)
		{
			float* h_ptr = &result.h_data[0].x;
			float* d_ptr = &result.d_data[0].x;
			CHECK(cudaMemcpy2D(h_ptr, sizeof(AkazePoint), d_ptr, sizeof(AkazePoint), (desc ? FLEN * sizeof(unsigned char) : 0) + 6 * sizeof(float), result.num_pts, cudaMemcpyDeviceToHost));
		}

		if (reused)
		{
			CHECK(cudaMemset(omem, 0, total_osize));
		}
		else
		{
			CHECK(cudaFree(tmem));
		}
	}


	void Akazer::allocMemory(void** addr, int3& whp0, int3* owhps, int* osizes, int* offsets, const bool reused)
	{
		owhps[0] = whp0;
		osizes[0] = whp0.y * whp0.z;
		offsets[0] = 3 * osizes[0];
		offsets[1] = offsets[0] + osizes[0] * max_scale * 4;
		for (int i = 0, j = 1, k = 2; j < noctaves; i++, j++, k++)
		{
			owhps[j].x = (owhps[i].x >> 1);
			owhps[j].y = (owhps[i].y >> 1);
			if (owhps[j].x < 80 || owhps[j].y < 80)
			{
				noctaves = j;
				break;
			}
			owhps[j].z = iAlignUp(owhps[j].x, 128);
			osizes[j] = owhps[j].y * owhps[j].z;
			offsets[k] = offsets[j] + osizes[j] * max_scale * 4;
		}

		if ((reused && !omem) || !reused)
		{
			CHECK(cudaMalloc(addr, offsets[noctaves] * sizeof(float)));
			CHECK(cudaMemset(*addr, 0, offsets[noctaves] * sizeof(float)));
		}

		if (reused)
		{
			total_osize = offsets[noctaves] * sizeof(float);
		}
	}


	void Akazer::detect(AkazeData& result, float* tmem, float* image, int3* owhps, int* osizes, int* offsets)
	{
		unsigned int* d_point_counter_addr;
		getPointCounter((void**)&d_point_counter_addr);
		CHECK(cudaMemset(d_point_counter_addr, 0, sizeof(unsigned int)));
		setMaxNumPoints(result.max_pts);

		int w, h, p, msz, ms_msz, mstep;
		float* response_map = tmem;
		float* size_map = tmem + osizes[0];
		int* layer_map = (int*)(size_map + osizes[0]);

		size_t nbytes = osizes[0] * sizeof(float);
		float minv = 1e-6f;
		int* iminv = (int*)&minv;
		CHECK(cudaMemset(layer_map, -1, nbytes));
		CHECK(cudaMemset(response_map, *iminv, nbytes));
		CHECK(cudaMemset(size_map, *iminv, nbytes));

		float* oldnld = NULL;
		float* nldimg = NULL;
		float* smooth = NULL;
		float* flow = NULL;
		float* temp = NULL;
		float* dx = NULL;
		float* dy = NULL;

		float tmax = 0.25f;
		float esigma = soffset;
		float last_etime = 0.5 * soffset * soffset;
		float curr_etime = 0;
		float ttime = 0;
		int naux = 0;
		int oratio = 1;
		int sigma_size = 0;

		float smax = 1.0f;
		if (FEATURE_TYPE == 0 || FEATURE_TYPE == 1 || FEATURE_TYPE == 4 || FEATURE_TYPE == 5)
		{
			smax = 10.0 * sqrtf(2.0f);
		}
		else if (FEATURE_TYPE == 2 || FEATURE_TYPE == 3)
		{
			smax = 12.0 * sqrtf(2.0f);
		}
		std::vector<float> exdata(max_scale * 2);
		float* borders = exdata.data();
		float* sizes = borders + max_scale;
		float psz = 10000;
		int neigh = 0;

		for (int i = 0; i < noctaves; i++)
		{
			w = owhps[i].x;
			h = owhps[i].y;
			p = owhps[i].z;
			msz = osizes[i];
			ms_msz = msz * max_scale;

			nldimg = tmem + offsets[i];
			smooth = nldimg + ms_msz;
			flow = smooth + ms_msz;
			temp = flow + ms_msz;
			dx = flow;
			dy = temp;

			for (int j = 0; j < max_scale; j++)
			{
				if (j == 0 && i == 0)
				{
					float var = soffset * soffset;
					int ksz = 2 * ceilf((soffset - 0.8f) / 0.3f) + 3;
					hLowPass(image, smooth, w, h, p, 1.f, 5);
					hScharrContrast(smooth, temp, kcontrast, per, w, h, p);
					hLowPass(image, nldimg, w, h, p, var, ksz);
					CHECK(cudaMemcpy(smooth, nldimg, msz * sizeof(float), cudaMemcpyDeviceToDevice));

					sizes[j] = esigma * derivative_factor;
					sigma_size = (int)(esigma * derivative_factor + 0.5f);
					borders[j] = smax * sigma_size;
					hHessianDeterminant(smooth, dx, dy, sigma_size, w, h, p);
					continue;
				}

				std::vector<float> tau;
				esigma = soffset * powf(2, (float)j / max_scale + i);
				curr_etime = 0.5f * esigma * esigma;
				ttime = curr_etime - last_etime;
				naux = fed_tau_by_process_time(ttime, 1, tmax, reordering, tau);
				sizes[j] = esigma * derivative_factor / oratio;
				sigma_size = (int)(sizes[j] + 0.5f);
				borders[j] = smax * sigma_size;

				if (j == 0)
				{
					kcontrast *= 0.75f;
					oldnld = nldimg - mstep;
					hDownWithSmooth(oldnld, nldimg, smooth, owhps[i - 1], owhps[i]);
					hFlow(smooth, flow, diffusivity, kcontrast, w, h, p);
					for (int k = 0; k < naux; k++)
					{
						hNldStep(nldimg, flow, temp, tau[k], w, h, p);
						CHECK(cudaMemcpy(nldimg, temp, msz * sizeof(float), cudaMemcpyDeviceToDevice));
					}
				}
				else
				{
					oldnld = nldimg;
					nldimg += msz;
					smooth += msz;
					flow += msz;
					temp += msz;
					dx = flow;
					dy = temp;

					hLowPass(oldnld, smooth, w, h, p, 1.f, 5);
					hFlow(smooth, flow, diffusivity, kcontrast, w, h, p);
					hNldStep(oldnld, flow, nldimg, tau[0], w, h, p);
					for (int k = 1; k < naux; k++)
					{
						hNldStep(nldimg, flow, temp, tau[k], w, h, p);
						CHECK(cudaMemcpy(nldimg, temp, msz * sizeof(float), cudaMemcpyDeviceToDevice));
					}
				}

				hHessianDeterminant(smooth, dx, dy, sigma_size, w, h, p);
				last_etime = curr_etime;
			}

			float* dets = tmem + offsets[i] + ms_msz;
			hCalcExtremaMap(dets, response_map, size_map, layer_map, borders, i, max_scale, dthreshold, w, h, p, owhps[0].z);
			psz = std::min(psz, borders[0] * oratio);
			neigh = std::max(neigh, sigma_size);

			mstep = ms_msz * 4;
			oratio *= 2;
		}

		hNmsR(result.d_data, response_map, size_map, layer_map, (int)psz, neigh, owhps[0].x, owhps[0].y, owhps[0].z);
		CHECK(cudaMemcpy(&result.num_pts, d_point_counter_addr, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		result.num_pts = std::min(result.num_pts, result.max_pts);

		sortAkazePoints(result.d_data, result.num_pts);

		setOparam(osizes, 5 * noctaves + 1);
		hRefine(result, tmem, noctaves, max_scale);
	}


	void Akazer::fastDetect(AkazeData& result, void* tmem, unsigned char* image, int3* owhps, int* osizes, int* offsets)
	{
		unsigned int* d_point_counter_addr;
		getPointCounter((void**)&d_point_counter_addr);
		CHECK(cudaMemset(d_point_counter_addr, 0, sizeof(unsigned int)));
		setMaxNumPoints(result.max_pts);

		int w, h, p, msz, ms_msz, mstep;
		int* response_map = (int*)tmem;
		float* size_map = (float*)response_map + osizes[0];
		int* layer_map = (int*)(size_map + osizes[0]);

		size_t nbytes = osizes[0] * sizeof(int);
		float minv = -1e6f;
		int* iminv = (int*)&minv;
		CHECK(cudaMemset(layer_map, -1, nbytes));
		CHECK(cudaMemset(response_map, 0x80, nbytes));
		CHECK(cudaMemset(size_map, *iminv, nbytes));

		int* oldnld = NULL;
		int* nldimg = NULL;
		int* smooth = NULL;
		int* flow = NULL;
		int* temp = NULL;
		int* dx = NULL;
		int* dy = NULL;

		float tmax = 0.25f;
		float esigma = soffset;
		float last_etime = 0.5 * soffset * soffset;
		float curr_etime = 0;
		float ttime = 0;
		int naux = 0;
		int oratio = 1;
		int sigma_size = 0;

		float smax = 1.0f;
		if (FEATURE_TYPE == 0 || FEATURE_TYPE == 1 || FEATURE_TYPE == 4 || FEATURE_TYPE == 5)
		{
			smax = 10.0 * sqrtf(2.0f);
		}
		else if (FEATURE_TYPE == 2 || FEATURE_TYPE == 3)
		{
			smax = 12.0 * sqrtf(2.0f);
		}
		std::vector<float> exdata(max_scale * 2);
		float* borders = exdata.data();
		float* sizes = borders + max_scale;
		float psz = 10000;
		int neigh = 0;

		int ikcontrast = 1;
		int idthreshold = 65;

		for (int i = 0; i < noctaves; i++)
		{
			w = owhps[i].x;
			h = owhps[i].y;
			p = owhps[i].z;
			msz = osizes[i];
			ms_msz = msz * max_scale;

			nldimg = (int*)tmem + offsets[i];
			smooth = nldimg + ms_msz;
			flow = smooth + ms_msz;
			temp = flow + ms_msz;
			dx = flow;
			dy = temp;

			for (int j = 0; j < max_scale; j++)
			{
				if (j == 0 && i == 0)
				{
					float var = soffset * soffset;
					int ksz = 2 * ceilf((soffset - 0.8f) / 0.3f) + 3;

					fastakaze::hConv2dR2(image, smooth, w, h, p, 1.f);
					fastakaze::hScharrContrast(smooth, temp, ikcontrast, per, w, h, p);
					fastakaze::hLowPass(image, nldimg, w, h, p, var, ksz);
					CHECK(cudaMemcpy(smooth, nldimg, msz * sizeof(int), cudaMemcpyDeviceToDevice));

					sizes[j] = esigma * derivative_factor;
					sigma_size = (int)(esigma * derivative_factor + 0.5f);
					borders[j] = smax * sigma_size;
					fastakaze::hHessianDeterminant(smooth, dx, dy, sigma_size, w, h, p);
					continue;
				}

				std::vector<float> tau;
				esigma = soffset * powf(2, (float)j / max_scale + i);
				curr_etime = 0.5f * esigma * esigma;
				ttime = curr_etime - last_etime;
				naux = fed_tau_by_process_time(ttime, 1, tmax, reordering, tau);
				sizes[j] = esigma * derivative_factor / oratio;
				sigma_size = (int)(sizes[j] + 0.5f);
				borders[j] = smax * sigma_size;

				if (j == 0)
				{
					ikcontrast = (int)(ikcontrast * 0.75f + 0.5f);
					oldnld = nldimg - mstep;
					fastakaze::hDownWithSmooth(oldnld, nldimg, smooth, owhps[i - 1], owhps[i]);
					fastakaze::hFlow(smooth, flow, diffusivity, ikcontrast, w, h, p);
					for (int k = 0; k < naux; k++)
					{
						fastakaze::hNldStep(nldimg, flow, temp, tau[k], w, h, p);
						CHECK(cudaMemcpy(nldimg, temp, msz * sizeof(int), cudaMemcpyDeviceToDevice));
					}
				}
				else
				{
					oldnld = nldimg;
					nldimg += msz;
					smooth += msz;
					flow += msz;
					temp += msz;
					dx = flow;
					dy = temp;

					fastakaze::hConv2dR2(oldnld, smooth, w, h, p, 1.f);
					fastakaze::hFlow(smooth, flow, diffusivity, ikcontrast, w, h, p);
					fastakaze::hNldStep(oldnld, flow, nldimg, tau[0], w, h, p);
					for (int k = 1; k < naux; k++)
					{
						fastakaze::hNldStep(nldimg, flow, temp, tau[k], w, h, p);
						CHECK(cudaMemcpy(nldimg, temp, msz * sizeof(int), cudaMemcpyDeviceToDevice));
					}
				}

				fastakaze::hHessianDeterminant(smooth, dx, dy, sigma_size, w, h, p);
				last_etime = curr_etime;
			}

			int* dets = (int*)tmem + offsets[i] + ms_msz;
			fastakaze::hCalcExtremaMap(dets, response_map, size_map, layer_map, borders, i, max_scale, idthreshold, w, h, p, owhps[0].z);
			psz = std::min(psz, borders[0] * oratio);
			neigh = std::max(neigh, sigma_size);

			mstep = ms_msz * 4;
			oratio *= 2;
		}

		fastakaze::hNmsR(result.d_data, response_map, size_map, layer_map, (int)psz, neigh, owhps[0].x, owhps[0].y, owhps[0].z);
		CHECK(cudaMemcpy(&result.num_pts, d_point_counter_addr, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		result.num_pts = std::min(result.num_pts, result.max_pts);

		sortAkazePoints(result.d_data, result.num_pts);

		setOparam(osizes, 5 * noctaves + 1);
		fastakaze::hRefine(result, tmem, noctaves, max_scale);
	}

}
