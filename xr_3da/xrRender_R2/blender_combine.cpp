#include "stdafx.h"
#pragma hdrstop

#include "Blender_combine.h"

CBlender_combine::CBlender_combine()
{
	description.CLS = 0;
}
CBlender_combine::~CBlender_combine()
{
}

void CBlender_combine::Compile(CBlender_Compile& C)
{
	IBlender::Compile(C);

	switch (C.iElement)
	{
	case 0: // combine
		C.r_Pass("combine_1", "combine_1", FALSE, FALSE, FALSE, TRUE, D3DBLEND_INVSRCALPHA,
				 D3DBLEND_SRCALPHA); //. MRT-blend?
		C.r_Sampler_rtf("s_accumulator", r2_RT_accum);
		C.r_Sampler_rtf("s_tonemap", r2_RT_luminance_cur);
		C.r_Sampler_clw("s_material", r2_material);
		C.r_Sampler_clf("env_s0", r2_T_envs0);
		C.r_Sampler_clf("env_s1", r2_T_envs1);
		C.r_Sampler_clf("sky_s0", r2_T_sky0);
		C.r_Sampler_clf("sky_s1", r2_T_sky1);
		gbuffer(C);
		C.r_End();
		break;
	case 1: // aa-edge-detection + AA :)
		C.r_Pass("null", "combine_2_AA", FALSE, FALSE, FALSE);
		C.r_Sampler_clf("s_image", r2_RT_generic0);
		C.r_Sampler_clf("s_bloom", r2_RT_bloom1);
		C.r_Sampler_clf("s_distort", r2_RT_generic1);
		gbuffer(C);
		C.r_End();
		break;
	case 2: // non-AA
		C.r_Pass("null", "combine_2_NAA", FALSE, FALSE, FALSE);
		C.r_Sampler_clf("s_image", r2_RT_generic0);
		C.r_Sampler_clf("s_bloom", r2_RT_bloom1);
		C.r_Sampler_clf("s_distort", r2_RT_generic1);
		gbuffer(C);
		C.r_End();
		break;
	case 3: // aa-edge-detection + AA :) + DISTORTION
		C.r_Pass("null", "combine_2_AA_D", FALSE, FALSE, FALSE);
		C.r_Sampler_clf("s_image", r2_RT_generic0);
		C.r_Sampler_clf("s_bloom", r2_RT_bloom1);
		C.r_Sampler_clf("s_distort", r2_RT_generic1);
		gbuffer(C);
		C.r_End();
		break;
	case 4: // non-AA + DISTORTION
		C.r_Pass("null", "combine_2_NAA_D", FALSE, FALSE, FALSE);
		C.r_Sampler_clf("s_image", r2_RT_generic0);
		C.r_Sampler_clf("s_bloom", r2_RT_bloom1);
		C.r_Sampler_clf("s_distort", r2_RT_generic1);
		gbuffer(C);
		C.r_End();
		break;
	case 5: // post-processing
		break;
	}
}
