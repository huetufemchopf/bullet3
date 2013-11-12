//this file is autogenerated using stringify.bat (premake --stringify) in the build folder of this project
static const char* solverUtilsCL= \
"/*\n"
"Copyright (c) 2013 Advanced Micro Devices, Inc.  \n"
"This software is provided 'as-is', without any express or implied warranty.\n"
"In no event will the authors be held liable for any damages arising from the use of this software.\n"
"Permission is granted to anyone to use this software for any purpose, \n"
"including commercial applications, and to alter it and redistribute it freely, \n"
"subject to the following restrictions:\n"
"1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.\n"
"2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.\n"
"3. This notice may not be removed or altered from any source distribution.\n"
"*/\n"
"//Originally written by Erwin Coumans\n"
"#ifndef B3_CONTACT4DATA_H\n"
"#define B3_CONTACT4DATA_H\n"
"#ifndef B3_FLOAT4_H\n"
"#define B3_FLOAT4_H\n"
"#ifndef B3_PLATFORM_DEFINITIONS_H\n"
"#define B3_PLATFORM_DEFINITIONS_H\n"
"struct MyTest\n"
"{\n"
"	int bla;\n"
"};\n"
"#ifdef __cplusplus\n"
"#else\n"
"#define b3AtomicInc atomic_inc\n"
"#define b3Fabs fabs\n"
"#define b3Sqrt native_sqrt\n"
"#define b3Sin native_sin\n"
"#define b3Cos native_cos\n"
"#endif\n"
"#endif\n"
"#ifdef __cplusplus\n"
"#else\n"
"	typedef float4	b3Float4;\n"
"	#define b3Float4ConstArg const b3Float4\n"
"	#define b3MakeFloat4 (float4)\n"
"	float b3Dot3F4(b3Float4ConstArg v0,b3Float4ConstArg v1)\n"
"	{\n"
"		float4 a1 = b3MakeFloat4(v0.xyz,0.f);\n"
"		float4 b1 = b3MakeFloat4(v1.xyz,0.f);\n"
"		return dot(a1, b1);\n"
"	}\n"
"	b3Float4 b3Cross3(b3Float4ConstArg v0,b3Float4ConstArg v1)\n"
"	{\n"
"		float4 a1 = b3MakeFloat4(v0.xyz,0.f);\n"
"		float4 b1 = b3MakeFloat4(v1.xyz,0.f);\n"
"		return cross(a1, b1);\n"
"	}\n"
"#endif \n"
"		\n"
"inline bool b3IsAlmostZero(b3Float4ConstArg v)\n"
"{\n"
"	if(b3Fabs(v.x)>1e-6 || b3Fabs(v.y)>1e-6 || b3Fabs(v.z)>1e-6)	\n"
"		return false;\n"
"	return true;\n"
"}\n"
"#endif //B3_FLOAT4_H\n"
"typedef  struct b3Contact4Data b3Contact4Data_t;\n"
"struct b3Contact4Data\n"
"{\n"
"	b3Float4	m_worldPosB[4];\n"
"//	b3Float4	m_localPosA[4];\n"
"//	b3Float4	m_localPosB[4];\n"
"	b3Float4	m_worldNormalOnB;	//	w: m_nPoints\n"
"	unsigned short  m_restituitionCoeffCmp;\n"
"	unsigned short  m_frictionCoeffCmp;\n"
"	int m_batchIdx;\n"
"	int m_bodyAPtrAndSignBit;//x:m_bodyAPtr, y:m_bodyBPtr\n"
"	int m_bodyBPtrAndSignBit;\n"
"	int	m_childIndexA;\n"
"	int	m_childIndexB;\n"
"	int m_unused1;\n"
"	int m_unused2;\n"
"};\n"
"inline int b3Contact4Data_getNumPoints(const struct b3Contact4Data* contact)\n"
"{\n"
"	return (int)contact->m_worldNormalOnB.w;\n"
"};\n"
"inline void b3Contact4Data_setNumPoints(struct b3Contact4Data* contact, int numPoints)\n"
"{\n"
"	contact->m_worldNormalOnB.w = (float)numPoints;\n"
"};\n"
"#endif //B3_CONTACT4DATA_H\n"
"#pragma OPENCL EXTENSION cl_amd_printf : enable\n"
"#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable\n"
"#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable\n"
"#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable\n"
"#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable\n"
"#ifdef cl_ext_atomic_counters_32\n"
"#pragma OPENCL EXTENSION cl_ext_atomic_counters_32 : enable\n"
"#else\n"
"#define counter32_t volatile global int*\n"
"#endif\n"
"typedef unsigned int u32;\n"
"typedef unsigned short u16;\n"
"typedef unsigned char u8;\n"
"#define GET_GROUP_IDX get_group_id(0)\n"
"#define GET_LOCAL_IDX get_local_id(0)\n"
"#define GET_GLOBAL_IDX get_global_id(0)\n"
"#define GET_GROUP_SIZE get_local_size(0)\n"
"#define GET_NUM_GROUPS get_num_groups(0)\n"
"#define GROUP_LDS_BARRIER barrier(CLK_LOCAL_MEM_FENCE)\n"
"#define GROUP_MEM_FENCE mem_fence(CLK_LOCAL_MEM_FENCE)\n"
"#define AtomInc(x) atom_inc(&(x))\n"
"#define AtomInc1(x, out) out = atom_inc(&(x))\n"
"#define AppendInc(x, out) out = atomic_inc(x)\n"
"#define AtomAdd(x, value) atom_add(&(x), value)\n"
"#define AtomCmpxhg(x, cmp, value) atom_cmpxchg( &(x), cmp, value )\n"
"#define AtomXhg(x, value) atom_xchg ( &(x), value )\n"
"#define SELECT_UINT4( b, a, condition ) select( b,a,condition )\n"
"#define make_float4 (float4)\n"
"#define make_float2 (float2)\n"
"#define make_uint4 (uint4)\n"
"#define make_int4 (int4)\n"
"#define make_uint2 (uint2)\n"
"#define make_int2 (int2)\n"
"#define max2 max\n"
"#define min2 min\n"
"///////////////////////////////////////\n"
"//	Vector\n"
"///////////////////////////////////////\n"
"__inline\n"
"float fastDiv(float numerator, float denominator)\n"
"{\n"
"	return native_divide(numerator, denominator);	\n"
"//	return numerator/denominator;	\n"
"}\n"
"__inline\n"
"float4 fastDiv4(float4 numerator, float4 denominator)\n"
"{\n"
"	return native_divide(numerator, denominator);	\n"
"}\n"
"__inline\n"
"float fastSqrtf(float f2)\n"
"{\n"
"	return native_sqrt(f2);\n"
"//	return sqrt(f2);\n"
"}\n"
"__inline\n"
"float fastRSqrt(float f2)\n"
"{\n"
"	return native_rsqrt(f2);\n"
"}\n"
"__inline\n"
"float fastLength4(float4 v)\n"
"{\n"
"	return fast_length(v);\n"
"}\n"
"__inline\n"
"float4 fastNormalize4(float4 v)\n"
"{\n"
"	return fast_normalize(v);\n"
"}\n"
"__inline\n"
"float sqrtf(float a)\n"
"{\n"
"//	return sqrt(a);\n"
"	return native_sqrt(a);\n"
"}\n"
"__inline\n"
"float4 cross3(float4 a1, float4 b1)\n"
"{\n"
"	float4 	a=make_float4(a1.xyz,0.f);\n"
"	float4 	b=make_float4(b1.xyz,0.f);\n"
"	//float4 	a=a1;\n"
"	//float4 	b=b1;\n"
"	return cross(a,b);\n"
"}\n"
"__inline\n"
"float dot3F4(float4 a, float4 b)\n"
"{\n"
"	float4 a1 = make_float4(a.xyz,0.f);\n"
"	float4 b1 = make_float4(b.xyz,0.f);\n"
"	return dot(a1, b1);\n"
"}\n"
"__inline\n"
"float length3(const float4 a)\n"
"{\n"
"	return sqrtf(dot3F4(a,a));\n"
"}\n"
"__inline\n"
"float dot4(const float4 a, const float4 b)\n"
"{\n"
"	return dot( a, b );\n"
"}\n"
"//	for height\n"
"__inline\n"
"float dot3w1(const float4 point, const float4 eqn)\n"
"{\n"
"	return dot3F4(point,eqn) + eqn.w;\n"
"}\n"
"__inline\n"
"float4 normalize3(const float4 a)\n"
"{\n"
"	float4 n = make_float4(a.x, a.y, a.z, 0.f);\n"
"	return fastNormalize4( n );\n"
"//	float length = sqrtf(dot3F4(a, a));\n"
"//	return 1.f/length * a;\n"
"}\n"
"__inline\n"
"float4 normalize4(const float4 a)\n"
"{\n"
"	float length = sqrtf(dot4(a, a));\n"
"	return 1.f/length * a;\n"
"}\n"
"__inline\n"
"float4 createEquation(const float4 a, const float4 b, const float4 c)\n"
"{\n"
"	float4 eqn;\n"
"	float4 ab = b-a;\n"
"	float4 ac = c-a;\n"
"	eqn = normalize3( cross3(ab, ac) );\n"
"	eqn.w = -dot3F4(eqn,a);\n"
"	return eqn;\n"
"}\n"
"///////////////////////////////////////\n"
"//	Matrix3x3\n"
"///////////////////////////////////////\n"
"typedef struct\n"
"{\n"
"	float4 m_row[3];\n"
"}Matrix3x3;\n"
"__inline\n"
"Matrix3x3 mtZero();\n"
"__inline\n"
"Matrix3x3 mtIdentity();\n"
"__inline\n"
"Matrix3x3 mtTranspose(Matrix3x3 m);\n"
"__inline\n"
"Matrix3x3 mtMul(Matrix3x3 a, Matrix3x3 b);\n"
"__inline\n"
"float4 mtMul1(Matrix3x3 a, float4 b);\n"
"__inline\n"
"float4 mtMul3(float4 a, Matrix3x3 b);\n"
"__inline\n"
"Matrix3x3 mtZero()\n"
"{\n"
"	Matrix3x3 m;\n"
"	m.m_row[0] = (float4)(0.f);\n"
"	m.m_row[1] = (float4)(0.f);\n"
"	m.m_row[2] = (float4)(0.f);\n"
"	return m;\n"
"}\n"
"__inline\n"
"Matrix3x3 mtIdentity()\n"
"{\n"
"	Matrix3x3 m;\n"
"	m.m_row[0] = (float4)(1,0,0,0);\n"
"	m.m_row[1] = (float4)(0,1,0,0);\n"
"	m.m_row[2] = (float4)(0,0,1,0);\n"
"	return m;\n"
"}\n"
"__inline\n"
"Matrix3x3 mtTranspose(Matrix3x3 m)\n"
"{\n"
"	Matrix3x3 out;\n"
"	out.m_row[0] = (float4)(m.m_row[0].x, m.m_row[1].x, m.m_row[2].x, 0.f);\n"
"	out.m_row[1] = (float4)(m.m_row[0].y, m.m_row[1].y, m.m_row[2].y, 0.f);\n"
"	out.m_row[2] = (float4)(m.m_row[0].z, m.m_row[1].z, m.m_row[2].z, 0.f);\n"
"	return out;\n"
"}\n"
"__inline\n"
"Matrix3x3 mtMul(Matrix3x3 a, Matrix3x3 b)\n"
"{\n"
"	Matrix3x3 transB;\n"
"	transB = mtTranspose( b );\n"
"	Matrix3x3 ans;\n"
"	//	why this doesn't run when 0ing in the for{}\n"
"	a.m_row[0].w = 0.f;\n"
"	a.m_row[1].w = 0.f;\n"
"	a.m_row[2].w = 0.f;\n"
"	for(int i=0; i<3; i++)\n"
"	{\n"
"//	a.m_row[i].w = 0.f;\n"
"		ans.m_row[i].x = dot3F4(a.m_row[i],transB.m_row[0]);\n"
"		ans.m_row[i].y = dot3F4(a.m_row[i],transB.m_row[1]);\n"
"		ans.m_row[i].z = dot3F4(a.m_row[i],transB.m_row[2]);\n"
"		ans.m_row[i].w = 0.f;\n"
"	}\n"
"	return ans;\n"
"}\n"
"__inline\n"
"float4 mtMul1(Matrix3x3 a, float4 b)\n"
"{\n"
"	float4 ans;\n"
"	ans.x = dot3F4( a.m_row[0], b );\n"
"	ans.y = dot3F4( a.m_row[1], b );\n"
"	ans.z = dot3F4( a.m_row[2], b );\n"
"	ans.w = 0.f;\n"
"	return ans;\n"
"}\n"
"__inline\n"
"float4 mtMul3(float4 a, Matrix3x3 b)\n"
"{\n"
"	float4 colx = make_float4(b.m_row[0].x, b.m_row[1].x, b.m_row[2].x, 0);\n"
"	float4 coly = make_float4(b.m_row[0].y, b.m_row[1].y, b.m_row[2].y, 0);\n"
"	float4 colz = make_float4(b.m_row[0].z, b.m_row[1].z, b.m_row[2].z, 0);\n"
"	float4 ans;\n"
"	ans.x = dot3F4( a, colx );\n"
"	ans.y = dot3F4( a, coly );\n"
"	ans.z = dot3F4( a, colz );\n"
"	return ans;\n"
"}\n"
"///////////////////////////////////////\n"
"//	Quaternion\n"
"///////////////////////////////////////\n"
"typedef float4 Quaternion;\n"
"__inline\n"
"Quaternion qtMul(Quaternion a, Quaternion b);\n"
"__inline\n"
"Quaternion qtNormalize(Quaternion in);\n"
"__inline\n"
"float4 qtRotate(Quaternion q, float4 vec);\n"
"__inline\n"
"Quaternion qtInvert(Quaternion q);\n"
"__inline\n"
"Quaternion qtMul(Quaternion a, Quaternion b)\n"
"{\n"
"	Quaternion ans;\n"
"	ans = cross3( a, b );\n"
"	ans += a.w*b+b.w*a;\n"
"//	ans.w = a.w*b.w - (a.x*b.x+a.y*b.y+a.z*b.z);\n"
"	ans.w = a.w*b.w - dot3F4(a, b);\n"
"	return ans;\n"
"}\n"
"__inline\n"
"Quaternion qtNormalize(Quaternion in)\n"
"{\n"
"	return fastNormalize4(in);\n"
"//	in /= length( in );\n"
"//	return in;\n"
"}\n"
"__inline\n"
"float4 qtRotate(Quaternion q, float4 vec)\n"
"{\n"
"	Quaternion qInv = qtInvert( q );\n"
"	float4 vcpy = vec;\n"
"	vcpy.w = 0.f;\n"
"	float4 out = qtMul(qtMul(q,vcpy),qInv);\n"
"	return out;\n"
"}\n"
"__inline\n"
"Quaternion qtInvert(Quaternion q)\n"
"{\n"
"	return (Quaternion)(-q.xyz, q.w);\n"
"}\n"
"__inline\n"
"float4 qtInvRotate(const Quaternion q, float4 vec)\n"
"{\n"
"	return qtRotate( qtInvert( q ), vec );\n"
"}\n"
"#define WG_SIZE 64\n"
"typedef struct\n"
"{\n"
"	float4 m_pos;\n"
"	Quaternion m_quat;\n"
"	float4 m_linVel;\n"
"	float4 m_angVel;\n"
"	u32 m_shapeIdx;\n"
"	float m_invMass;\n"
"	float m_restituitionCoeff;\n"
"	float m_frictionCoeff;\n"
"} Body;\n"
"typedef struct\n"
"{\n"
"	Matrix3x3 m_invInertia;\n"
"	Matrix3x3 m_initInvInertia;\n"
"} Shape;\n"
"typedef struct\n"
"{\n"
"	float4 m_linear;\n"
"	float4 m_worldPos[4];\n"
"	float4 m_center;	\n"
"	float m_jacCoeffInv[4];\n"
"	float m_b[4];\n"
"	float m_appliedRambdaDt[4];\n"
"	float m_fJacCoeffInv[2];	\n"
"	float m_fAppliedRambdaDt[2];	\n"
"	u32 m_bodyA;\n"
"	u32 m_bodyB;\n"
"	int m_batchIdx;\n"
"	u32 m_paddings;\n"
"} Constraint4;\n"
"__kernel void CountBodiesKernel(__global struct b3Contact4Data* manifoldPtr, __global unsigned int* bodyCount, __global int2* contactConstraintOffsets, int numContactManifolds, int fixedBodyIndex)\n"
"{\n"
"	int i = GET_GLOBAL_IDX;\n"
"	\n"
"	if( i < numContactManifolds)\n"
"	{\n"
"		int pa = manifoldPtr[i].m_bodyAPtrAndSignBit;\n"
"		bool isFixedA = (pa <0) || (pa == fixedBodyIndex);\n"
"		int bodyIndexA = abs(pa);\n"
"		if (!isFixedA)\n"
"		{\n"
"			 AtomInc1(bodyCount[bodyIndexA],contactConstraintOffsets[i].x);\n"
"		}\n"
"		barrier(CLK_GLOBAL_MEM_FENCE);\n"
"		int pb = manifoldPtr[i].m_bodyBPtrAndSignBit;\n"
"		bool isFixedB = (pb <0) || (pb == fixedBodyIndex);\n"
"		int bodyIndexB = abs(pb);\n"
"		if (!isFixedB)\n"
"		{\n"
"			AtomInc1(bodyCount[bodyIndexB],contactConstraintOffsets[i].y);\n"
"		} \n"
"	}\n"
"}\n"
"__kernel void ClearVelocitiesKernel(__global float4* linearVelocities,__global float4* angularVelocities, int numSplitBodies)\n"
"{\n"
"	int i = GET_GLOBAL_IDX;\n"
"	\n"
"	if( i < numSplitBodies)\n"
"	{\n"
"		linearVelocities[i] = make_float4(0);\n"
"		angularVelocities[i] = make_float4(0);\n"
"	}\n"
"}\n"
"__kernel void AverageVelocitiesKernel(__global Body* gBodies,__global int* offsetSplitBodies,__global const unsigned int* bodyCount,\n"
"__global float4* deltaLinearVelocities, __global float4* deltaAngularVelocities, int numBodies)\n"
"{\n"
"	int i = GET_GLOBAL_IDX;\n"
"	if (i<numBodies)\n"
"	{\n"
"		if (gBodies[i].m_invMass)\n"
"		{\n"
"			int bodyOffset = offsetSplitBodies[i];\n"
"			int count = bodyCount[i];\n"
"			float factor = 1.f/((float)count);\n"
"			float4 averageLinVel = make_float4(0.f);\n"
"			float4 averageAngVel = make_float4(0.f);\n"
"			\n"
"			for (int j=0;j<count;j++)\n"
"			{\n"
"				averageLinVel += deltaLinearVelocities[bodyOffset+j]*factor;\n"
"				averageAngVel += deltaAngularVelocities[bodyOffset+j]*factor;\n"
"			}\n"
"			\n"
"			for (int j=0;j<count;j++)\n"
"			{\n"
"				deltaLinearVelocities[bodyOffset+j] = averageLinVel;\n"
"				deltaAngularVelocities[bodyOffset+j] = averageAngVel;\n"
"			}\n"
"			\n"
"		}//bodies[i].m_invMass\n"
"	}//i<numBodies\n"
"}\n"
"void setLinearAndAngular( float4 n, float4 r0, float4 r1, float4* linear, float4* angular0, float4* angular1)\n"
"{\n"
"	*linear = make_float4(n.xyz,0.f);\n"
"	*angular0 = cross3(r0, n);\n"
"	*angular1 = -cross3(r1, n);\n"
"}\n"
"float calcRelVel( float4 l0, float4 l1, float4 a0, float4 a1, float4 linVel0, float4 angVel0, float4 linVel1, float4 angVel1 )\n"
"{\n"
"	return dot3F4(l0, linVel0) + dot3F4(a0, angVel0) + dot3F4(l1, linVel1) + dot3F4(a1, angVel1);\n"
"}\n"
"float calcJacCoeff(const float4 linear0, const float4 linear1, const float4 angular0, const float4 angular1,\n"
"					float invMass0, const Matrix3x3* invInertia0, float invMass1, const Matrix3x3* invInertia1, float countA, float countB)\n"
"{\n"
"	//	linear0,1 are normlized\n"
"	float jmj0 = invMass0;//dot3F4(linear0, linear0)*invMass0;\n"
"	float jmj1 = dot3F4(mtMul3(angular0,*invInertia0), angular0);\n"
"	float jmj2 = invMass1;//dot3F4(linear1, linear1)*invMass1;\n"
"	float jmj3 = dot3F4(mtMul3(angular1,*invInertia1), angular1);\n"
"	return -1.f/((jmj0+jmj1)*countA+(jmj2+jmj3)*countB);\n"
"}\n"
"void btPlaneSpace1 (float4 n, float4* p, float4* q);\n"
" void btPlaneSpace1 (float4 n, float4* p, float4* q)\n"
"{\n"
"  if (fabs(n.z) > 0.70710678f) {\n"
"    // choose p in y-z plane\n"
"    float a = n.y*n.y + n.z*n.z;\n"
"    float k = 1.f/sqrt(a);\n"
"    p[0].x = 0;\n"
"	p[0].y = -n.z*k;\n"
"	p[0].z = n.y*k;\n"
"    // set q = n x p\n"
"    q[0].x = a*k;\n"
"	q[0].y = -n.x*p[0].z;\n"
"	q[0].z = n.x*p[0].y;\n"
"  }\n"
"  else {\n"
"    // choose p in x-y plane\n"
"    float a = n.x*n.x + n.y*n.y;\n"
"    float k = 1.f/sqrt(a);\n"
"    p[0].x = -n.y*k;\n"
"	p[0].y = n.x*k;\n"
"	p[0].z = 0;\n"
"    // set q = n x p\n"
"    q[0].x = -n.z*p[0].y;\n"
"	q[0].y = n.z*p[0].x;\n"
"	q[0].z = a*k;\n"
"  }\n"
"}\n"
"void solveContact(__global Constraint4* cs,\n"
"			float4 posA, float4* linVelA, float4* angVelA, float invMassA, Matrix3x3 invInertiaA,\n"
"			float4 posB, float4* linVelB, float4* angVelB, float invMassB, Matrix3x3 invInertiaB,\n"
"			float4* dLinVelA, float4* dAngVelA, float4* dLinVelB, float4* dAngVelB)\n"
"{\n"
"	float minRambdaDt = 0;\n"
"	float maxRambdaDt = FLT_MAX;\n"
"	for(int ic=0; ic<4; ic++)\n"
"	{\n"
"		if( cs->m_jacCoeffInv[ic] == 0.f ) continue;\n"
"		float4 angular0, angular1, linear;\n"
"		float4 r0 = cs->m_worldPos[ic] - posA;\n"
"		float4 r1 = cs->m_worldPos[ic] - posB;\n"
"		setLinearAndAngular( cs->m_linear, r0, r1, &linear, &angular0, &angular1 );\n"
"	\n"
"		float rambdaDt = calcRelVel( cs->m_linear, -cs->m_linear, angular0, angular1, \n"
"			*linVelA+*dLinVelA, *angVelA+*dAngVelA, *linVelB+*dLinVelB, *angVelB+*dAngVelB ) + cs->m_b[ic];\n"
"		rambdaDt *= cs->m_jacCoeffInv[ic];\n"
"		\n"
"		{\n"
"			float prevSum = cs->m_appliedRambdaDt[ic];\n"
"			float updated = prevSum;\n"
"			updated += rambdaDt;\n"
"			updated = max2( updated, minRambdaDt );\n"
"			updated = min2( updated, maxRambdaDt );\n"
"			rambdaDt = updated - prevSum;\n"
"			cs->m_appliedRambdaDt[ic] = updated;\n"
"		}\n"
"			\n"
"		float4 linImp0 = invMassA*linear*rambdaDt;\n"
"		float4 linImp1 = invMassB*(-linear)*rambdaDt;\n"
"		float4 angImp0 = mtMul1(invInertiaA, angular0)*rambdaDt;\n"
"		float4 angImp1 = mtMul1(invInertiaB, angular1)*rambdaDt;\n"
"		\n"
"		if (invMassA)\n"
"		{\n"
"			*dLinVelA += linImp0;\n"
"			*dAngVelA += angImp0;\n"
"		}\n"
"		if (invMassB)\n"
"		{\n"
"			*dLinVelB += linImp1;\n"
"			*dAngVelB += angImp1;\n"
"		}\n"
"	}\n"
"}\n"
"//	solveContactConstraint( gBodies, gShapes, &gConstraints[i] ,contactConstraintOffsets,offsetSplitBodies, deltaLinearVelocities, deltaAngularVelocities);\n"
"void solveContactConstraint(__global Body* gBodies, __global Shape* gShapes, __global Constraint4* ldsCs, \n"
"__global int2* contactConstraintOffsets,__global unsigned int* offsetSplitBodies,\n"
"__global float4* deltaLinearVelocities, __global float4* deltaAngularVelocities)\n"
"{\n"
"	//float frictionCoeff = ldsCs[0].m_linear.w;\n"
"	int aIdx = ldsCs[0].m_bodyA;\n"
"	int bIdx = ldsCs[0].m_bodyB;\n"
"	float4 posA = gBodies[aIdx].m_pos;\n"
"	float4 linVelA = gBodies[aIdx].m_linVel;\n"
"	float4 angVelA = gBodies[aIdx].m_angVel;\n"
"	float invMassA = gBodies[aIdx].m_invMass;\n"
"	Matrix3x3 invInertiaA = gShapes[aIdx].m_invInertia;\n"
"	float4 posB = gBodies[bIdx].m_pos;\n"
"	float4 linVelB = gBodies[bIdx].m_linVel;\n"
"	float4 angVelB = gBodies[bIdx].m_angVel;\n"
"	float invMassB = gBodies[bIdx].m_invMass;\n"
"	Matrix3x3 invInertiaB = gShapes[bIdx].m_invInertia;\n"
"			\n"
"	float4 dLinVelA = make_float4(0,0,0,0);\n"
"	float4 dAngVelA = make_float4(0,0,0,0);\n"
"	float4 dLinVelB = make_float4(0,0,0,0);\n"
"	float4 dAngVelB = make_float4(0,0,0,0);\n"
"			\n"
"	int bodyOffsetA = offsetSplitBodies[aIdx];\n"
"	int constraintOffsetA = contactConstraintOffsets[0].x;\n"
"	int splitIndexA = bodyOffsetA+constraintOffsetA;\n"
"	\n"
"	if (invMassA)\n"
"	{\n"
"		dLinVelA = deltaLinearVelocities[splitIndexA];\n"
"		dAngVelA = deltaAngularVelocities[splitIndexA];\n"
"	}\n"
"	int bodyOffsetB = offsetSplitBodies[bIdx];\n"
"	int constraintOffsetB = contactConstraintOffsets[0].y;\n"
"	int splitIndexB= bodyOffsetB+constraintOffsetB;\n"
"	if (invMassB)\n"
"	{\n"
"		dLinVelB = deltaLinearVelocities[splitIndexB];\n"
"		dAngVelB = deltaAngularVelocities[splitIndexB];\n"
"	}\n"
"	solveContact( ldsCs, posA, &linVelA, &angVelA, invMassA, invInertiaA,\n"
"			posB, &linVelB, &angVelB, invMassB, invInertiaB ,&dLinVelA, &dAngVelA, &dLinVelB, &dAngVelB);\n"
"	if (invMassA)\n"
"	{\n"
"		deltaLinearVelocities[splitIndexA] = dLinVelA;\n"
"		deltaAngularVelocities[splitIndexA] = dAngVelA;\n"
"	} \n"
"	if (invMassB)\n"
"	{\n"
"		deltaLinearVelocities[splitIndexB] = dLinVelB;\n"
"		deltaAngularVelocities[splitIndexB] = dAngVelB;\n"
"	}\n"
"}\n"
"__kernel void SolveContactJacobiKernel(__global Constraint4* gConstraints, __global Body* gBodies, __global Shape* gShapes ,\n"
"__global int2* contactConstraintOffsets,__global unsigned int* offsetSplitBodies,__global float4* deltaLinearVelocities, __global float4* deltaAngularVelocities,\n"
"float deltaTime, float positionDrift, float positionConstraintCoeff, int fixedBodyIndex, int numManifolds\n"
")\n"
"{\n"
"	int i = GET_GLOBAL_IDX;\n"
"	if (i<numManifolds)\n"
"	{\n"
"		solveContactConstraint( gBodies, gShapes, &gConstraints[i] ,&contactConstraintOffsets[i],offsetSplitBodies, deltaLinearVelocities, deltaAngularVelocities);\n"
"	}\n"
"}\n"
"void solveFrictionConstraint(__global Body* gBodies, __global Shape* gShapes, __global Constraint4* ldsCs,\n"
"							__global int2* contactConstraintOffsets,__global unsigned int* offsetSplitBodies,\n"
"							__global float4* deltaLinearVelocities, __global float4* deltaAngularVelocities)\n"
"{\n"
"	float frictionCoeff = 0.7f;//ldsCs[0].m_linear.w;\n"
"	int aIdx = ldsCs[0].m_bodyA;\n"
"	int bIdx = ldsCs[0].m_bodyB;\n"
"	float4 posA = gBodies[aIdx].m_pos;\n"
"	float4 linVelA = gBodies[aIdx].m_linVel;\n"
"	float4 angVelA = gBodies[aIdx].m_angVel;\n"
"	float invMassA = gBodies[aIdx].m_invMass;\n"
"	Matrix3x3 invInertiaA = gShapes[aIdx].m_invInertia;\n"
"	float4 posB = gBodies[bIdx].m_pos;\n"
"	float4 linVelB = gBodies[bIdx].m_linVel;\n"
"	float4 angVelB = gBodies[bIdx].m_angVel;\n"
"	float invMassB = gBodies[bIdx].m_invMass;\n"
"	Matrix3x3 invInertiaB = gShapes[bIdx].m_invInertia;\n"
"	\n"
"	float4 dLinVelA = make_float4(0,0,0,0);\n"
"	float4 dAngVelA = make_float4(0,0,0,0);\n"
"	float4 dLinVelB = make_float4(0,0,0,0);\n"
"	float4 dAngVelB = make_float4(0,0,0,0);\n"
"			\n"
"	int bodyOffsetA = offsetSplitBodies[aIdx];\n"
"	int constraintOffsetA = contactConstraintOffsets[0].x;\n"
"	int splitIndexA = bodyOffsetA+constraintOffsetA;\n"
"	\n"
"	if (invMassA)\n"
"	{\n"
"		dLinVelA = deltaLinearVelocities[splitIndexA];\n"
"		dAngVelA = deltaAngularVelocities[splitIndexA];\n"
"	}\n"
"	int bodyOffsetB = offsetSplitBodies[bIdx];\n"
"	int constraintOffsetB = contactConstraintOffsets[0].y;\n"
"	int splitIndexB= bodyOffsetB+constraintOffsetB;\n"
"	if (invMassB)\n"
"	{\n"
"		dLinVelB = deltaLinearVelocities[splitIndexB];\n"
"		dAngVelB = deltaAngularVelocities[splitIndexB];\n"
"	}\n"
"	{\n"
"		float maxRambdaDt[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};\n"
"		float minRambdaDt[4] = {0.f,0.f,0.f,0.f};\n"
"		float sum = 0;\n"
"		for(int j=0; j<4; j++)\n"
"		{\n"
"			sum +=ldsCs[0].m_appliedRambdaDt[j];\n"
"		}\n"
"		frictionCoeff = 0.7f;\n"
"		for(int j=0; j<4; j++)\n"
"		{\n"
"			maxRambdaDt[j] = frictionCoeff*sum;\n"
"			minRambdaDt[j] = -maxRambdaDt[j];\n"
"		}\n"
"		\n"
"//		solveFriction( ldsCs, posA, &linVelA, &angVelA, invMassA, invInertiaA,\n"
"//			posB, &linVelB, &angVelB, invMassB, invInertiaB, maxRambdaDt, minRambdaDt );\n"
"		\n"
"		\n"
"		{\n"
"			\n"
"			__global Constraint4* cs = ldsCs;\n"
"			\n"
"			if( cs->m_fJacCoeffInv[0] == 0 && cs->m_fJacCoeffInv[0] == 0 ) return;\n"
"			const float4 center = cs->m_center;\n"
"			\n"
"			float4 n = -cs->m_linear;\n"
"			\n"
"			float4 tangent[2];\n"
"			btPlaneSpace1(n,&tangent[0],&tangent[1]);\n"
"			float4 angular0, angular1, linear;\n"
"			float4 r0 = center - posA;\n"
"			float4 r1 = center - posB;\n"
"			for(int i=0; i<2; i++)\n"
"			{\n"
"				setLinearAndAngular( tangent[i], r0, r1, &linear, &angular0, &angular1 );\n"
"				float rambdaDt = calcRelVel(linear, -linear, angular0, angular1,\n"
"											linVelA+dLinVelA, angVelA+dAngVelA, linVelB+dLinVelB, angVelB+dAngVelB );\n"
"				rambdaDt *= cs->m_fJacCoeffInv[i];\n"
"				\n"
"				{\n"
"					float prevSum = cs->m_fAppliedRambdaDt[i];\n"
"					float updated = prevSum;\n"
"					updated += rambdaDt;\n"
"					updated = max2( updated, minRambdaDt[i] );\n"
"					updated = min2( updated, maxRambdaDt[i] );\n"
"					rambdaDt = updated - prevSum;\n"
"					cs->m_fAppliedRambdaDt[i] = updated;\n"
"				}\n"
"				\n"
"				float4 linImp0 = invMassA*linear*rambdaDt;\n"
"				float4 linImp1 = invMassB*(-linear)*rambdaDt;\n"
"				float4 angImp0 = mtMul1(invInertiaA, angular0)*rambdaDt;\n"
"				float4 angImp1 = mtMul1(invInertiaB, angular1)*rambdaDt;\n"
"				\n"
"				dLinVelA += linImp0;\n"
"				dAngVelA += angImp0;\n"
"				dLinVelB += linImp1;\n"
"				dAngVelB += angImp1;\n"
"			}\n"
"			{	//	angular damping for point constraint\n"
"				float4 ab = normalize3( posB - posA );\n"
"				float4 ac = normalize3( center - posA );\n"
"				if( dot3F4( ab, ac ) > 0.95f  || (invMassA == 0.f || invMassB == 0.f))\n"
"				{\n"
"					float angNA = dot3F4( n, angVelA );\n"
"					float angNB = dot3F4( n, angVelB );\n"
"					\n"
"					dAngVelA -= (angNA*0.1f)*n;\n"
"					dAngVelB -= (angNB*0.1f)*n;\n"
"				}\n"
"			}\n"
"		}\n"
"		\n"
"		\n"
"	}\n"
"	if (invMassA)\n"
"	{\n"
"		deltaLinearVelocities[splitIndexA] = dLinVelA;\n"
"		deltaAngularVelocities[splitIndexA] = dAngVelA;\n"
"	} \n"
"	if (invMassB)\n"
"	{\n"
"		deltaLinearVelocities[splitIndexB] = dLinVelB;\n"
"		deltaAngularVelocities[splitIndexB] = dAngVelB;\n"
"	}\n"
" \n"
"}\n"
"__kernel void SolveFrictionJacobiKernel(__global Constraint4* gConstraints, __global Body* gBodies, __global Shape* gShapes ,\n"
"										__global int2* contactConstraintOffsets,__global unsigned int* offsetSplitBodies,\n"
"										__global float4* deltaLinearVelocities, __global float4* deltaAngularVelocities,\n"
"										float deltaTime, float positionDrift, float positionConstraintCoeff, int fixedBodyIndex, int numManifolds\n"
")\n"
"{\n"
"	int i = GET_GLOBAL_IDX;\n"
"	if (i<numManifolds)\n"
"	{\n"
"		solveFrictionConstraint( gBodies, gShapes, &gConstraints[i] ,&contactConstraintOffsets[i],offsetSplitBodies, deltaLinearVelocities, deltaAngularVelocities);\n"
"	}\n"
"}\n"
"__kernel void UpdateBodyVelocitiesKernel(__global Body* gBodies,__global int* offsetSplitBodies,__global const unsigned int* bodyCount,\n"
"									__global float4* deltaLinearVelocities, __global float4* deltaAngularVelocities, int numBodies)\n"
"{\n"
"	int i = GET_GLOBAL_IDX;\n"
"	if (i<numBodies)\n"
"	{\n"
"		if (gBodies[i].m_invMass)\n"
"		{\n"
"			int bodyOffset = offsetSplitBodies[i];\n"
"			int count = bodyCount[i];\n"
"			if (count)\n"
"			{\n"
"				gBodies[i].m_linVel += deltaLinearVelocities[bodyOffset];\n"
"				gBodies[i].m_angVel += deltaAngularVelocities[bodyOffset];\n"
"			}\n"
"		}\n"
"	}\n"
"}\n"
"void setConstraint4( const float4 posA, const float4 linVelA, const float4 angVelA, float invMassA, const Matrix3x3 invInertiaA,\n"
"	const float4 posB, const float4 linVelB, const float4 angVelB, float invMassB, const Matrix3x3 invInertiaB, \n"
"	__global struct b3Contact4Data* src, float dt, float positionDrift, float positionConstraintCoeff,float countA, float countB,\n"
"	Constraint4* dstC )\n"
"{\n"
"	dstC->m_bodyA = abs(src->m_bodyAPtrAndSignBit);\n"
"	dstC->m_bodyB = abs(src->m_bodyBPtrAndSignBit);\n"
"	float dtInv = 1.f/dt;\n"
"	for(int ic=0; ic<4; ic++)\n"
"	{\n"
"		dstC->m_appliedRambdaDt[ic] = 0.f;\n"
"	}\n"
"	dstC->m_fJacCoeffInv[0] = dstC->m_fJacCoeffInv[1] = 0.f;\n"
"	dstC->m_linear = src->m_worldNormalOnB;\n"
"	dstC->m_linear.w = 0.7f ;//src->getFrictionCoeff() );\n"
"	for(int ic=0; ic<4; ic++)\n"
"	{\n"
"		float4 r0 = src->m_worldPosB[ic] - posA;\n"
"		float4 r1 = src->m_worldPosB[ic] - posB;\n"
"		if( ic >= src->m_worldNormalOnB.w )//npoints\n"
"		{\n"
"			dstC->m_jacCoeffInv[ic] = 0.f;\n"
"			continue;\n"
"		}\n"
"		float relVelN;\n"
"		{\n"
"			float4 linear, angular0, angular1;\n"
"			setLinearAndAngular(src->m_worldNormalOnB, r0, r1, &linear, &angular0, &angular1);\n"
"			dstC->m_jacCoeffInv[ic] = calcJacCoeff(linear, -linear, angular0, angular1,\n"
"				invMassA, &invInertiaA, invMassB, &invInertiaB , countA, countB);\n"
"			relVelN = calcRelVel(linear, -linear, angular0, angular1,\n"
"				linVelA, angVelA, linVelB, angVelB);\n"
"			float e = 0.f;//src->getRestituitionCoeff();\n"
"			if( relVelN*relVelN < 0.004f ) e = 0.f;\n"
"			dstC->m_b[ic] = e*relVelN;\n"
"			//float penetration = src->m_worldPosB[ic].w;\n"
"			dstC->m_b[ic] += (src->m_worldPosB[ic].w + positionDrift)*positionConstraintCoeff*dtInv;\n"
"			dstC->m_appliedRambdaDt[ic] = 0.f;\n"
"		}\n"
"	}\n"
"	if( src->m_worldNormalOnB.w > 0 )//npoints\n"
"	{	//	prepare friction\n"
"		float4 center = make_float4(0.f);\n"
"		for(int i=0; i<src->m_worldNormalOnB.w; i++) \n"
"			center += src->m_worldPosB[i];\n"
"		center /= (float)src->m_worldNormalOnB.w;\n"
"		float4 tangent[2];\n"
"		btPlaneSpace1(-src->m_worldNormalOnB,&tangent[0],&tangent[1]);\n"
"		\n"
"		float4 r[2];\n"
"		r[0] = center - posA;\n"
"		r[1] = center - posB;\n"
"		for(int i=0; i<2; i++)\n"
"		{\n"
"			float4 linear, angular0, angular1;\n"
"			setLinearAndAngular(tangent[i], r[0], r[1], &linear, &angular0, &angular1);\n"
"			dstC->m_fJacCoeffInv[i] = calcJacCoeff(linear, -linear, angular0, angular1,\n"
"				invMassA, &invInertiaA, invMassB, &invInertiaB ,countA, countB);\n"
"			dstC->m_fAppliedRambdaDt[i] = 0.f;\n"
"		}\n"
"		dstC->m_center = center;\n"
"	}\n"
"	for(int i=0; i<4; i++)\n"
"	{\n"
"		if( i<src->m_worldNormalOnB.w )\n"
"		{\n"
"			dstC->m_worldPos[i] = src->m_worldPosB[i];\n"
"		}\n"
"		else\n"
"		{\n"
"			dstC->m_worldPos[i] = make_float4(0.f);\n"
"		}\n"
"	}\n"
"}\n"
"__kernel\n"
"__attribute__((reqd_work_group_size(WG_SIZE,1,1)))\n"
"void ContactToConstraintSplitKernel(__global const struct b3Contact4Data* gContact, __global const Body* gBodies, __global const Shape* gShapes, __global Constraint4* gConstraintOut, \n"
"__global const unsigned int* bodyCount,\n"
"int nContacts,\n"
"float dt,\n"
"float positionDrift,\n"
"float positionConstraintCoeff\n"
")\n"
"{\n"
"	int gIdx = GET_GLOBAL_IDX;\n"
"	\n"
"	if( gIdx < nContacts )\n"
"	{\n"
"		int aIdx = abs(gContact[gIdx].m_bodyAPtrAndSignBit);\n"
"		int bIdx = abs(gContact[gIdx].m_bodyBPtrAndSignBit);\n"
"		float4 posA = gBodies[aIdx].m_pos;\n"
"		float4 linVelA = gBodies[aIdx].m_linVel;\n"
"		float4 angVelA = gBodies[aIdx].m_angVel;\n"
"		float invMassA = gBodies[aIdx].m_invMass;\n"
"		Matrix3x3 invInertiaA = gShapes[aIdx].m_invInertia;\n"
"		float4 posB = gBodies[bIdx].m_pos;\n"
"		float4 linVelB = gBodies[bIdx].m_linVel;\n"
"		float4 angVelB = gBodies[bIdx].m_angVel;\n"
"		float invMassB = gBodies[bIdx].m_invMass;\n"
"		Matrix3x3 invInertiaB = gShapes[bIdx].m_invInertia;\n"
"		Constraint4 cs;\n"
"		float countA = invMassA ? (float)bodyCount[aIdx] : 1;\n"
"		float countB = invMassB ? (float)bodyCount[bIdx] : 1;\n"
"    	setConstraint4( posA, linVelA, angVelA, invMassA, invInertiaA, posB, linVelB, angVelB, invMassB, invInertiaB,\n"
"			&gContact[gIdx], dt, positionDrift, positionConstraintCoeff,countA,countB,\n"
"			&cs  );\n"
"		\n"
"		cs.m_batchIdx = gContact[gIdx].m_batchIdx;\n"
"		gConstraintOut[gIdx] = cs;\n"
"	}\n"
"}\n"
;