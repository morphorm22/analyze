#pragma once

#include "Tri6.hpp"
#include "PlatoMathTypes.hpp"

namespace Plato {

/******************************************************************************/
/*! Tet10 Element

    See Klaus-Jurgen Bathe, "Finite Element Proceedures", 2006, pg 375.
*/
/******************************************************************************/
class Tet10
{
  public:

    using Face = Plato::Tri6;

    static constexpr Plato::OrdinalType mNumSpatialDims  = 3;
    static constexpr Plato::OrdinalType mNumNodesPerCell = 10;
    static constexpr Plato::OrdinalType mNumNodesPerFace = 6;

    static constexpr Plato::OrdinalType mNumSpatialDimsOnFace = mNumSpatialDims-1;

//    static inline Plato::Array<1>
//    getCubWeights() { return Plato::Array<1>({Plato::Scalar(1.0)/6.0}); }

    static inline Plato::Array<4>
    getCubWeights() { return Plato::Array<4>({Plato::Scalar(1.0)/24.0, Plato::Scalar(1.0)/24.0, Plato::Scalar(1.0)/24.0, Plato::Scalar(1.0)/24.0}); }

//    static inline Plato::Array<5>
//    getCubWeights() { return Plato::Array<5>({Plato::Scalar(-2)/15, Plato::Scalar(3)/40, Plato::Scalar(3)/40, Plato::Scalar(3)/40, Plato::Scalar(3)/40}); }

//    static inline Plato::Array<14>
//    getCubWeights() { return Plato::Array<14>(
//      {1.2248840519393658257285034247721250587182e-2,
//       1.2248840519393658257285034247721250587182e-2,
//       1.2248840519393658257285034247721250587182e-2,
//       1.2248840519393658257285034247721250587182e-2,
//       1.8781320953002641799864275388881055635078e-2,
//       1.8781320953002641799864275388881055635078e-2,
//       1.8781320953002641799864275388881055635078e-2,
//       1.8781320953002641799864275388881055635078e-2,
//       7.0910034628469110730115713533762402962712e-3,
//       7.0910034628469110730115713533762402962712e-3,
//       7.0910034628469110730115713533762402962712e-3,
//       7.0910034628469110730115713533762402962712e-3,
//       7.0910034628469110730115713533762402962712e-3,
//       7.0910034628469110730115713533762402962712e-3}); }

//       {{1./4., 1./4., 1./4.},
//        {1./6., 1./6., 1./6.},
//        {1./6., 1./6., 1./2.},
//        {1./6., 1./2., 1./6.},
//        {1./2., 1./6., 1./6.}},
//       {-2./15.,
//       3./40.,
//        3./40.,
//        3./40.,
//        3./40.}

//    static inline Plato::Matrix<1,mNumSpatialDims>
//    getCubPoints()
//    {
//        return Plato::Matrix<1,mNumSpatialDims>({
//            Plato::Scalar(1)/4, Plato::Scalar(1)/4, Plato::Scalar(1)/4
//        });
//    }

    static inline Plato::Matrix<4,mNumSpatialDims>
    getCubPoints()
    {
        return Plato::Matrix<4,mNumSpatialDims>({
            0.585410196624969, 0.138196601125011, 0.138196601125011,
            0.138196601125011, 0.585410196624969, 0.138196601125011,
            0.138196601125011, 0.138196601125011, 0.585410196624969,
            0.138196601125011, 0.138196601125011, 0.138196601125011
        });
    }

//    static inline Plato::Matrix<5,mNumSpatialDims>
//    getCubPoints()
//    {
//        return Plato::Matrix<5,mNumSpatialDims>({
//            Plato::Scalar(1)/4, Plato::Scalar(1)/4, Plato::Scalar(1)/4,
//            Plato::Scalar(1)/6, Plato::Scalar(1)/6, Plato::Scalar(1)/6,
//            Plato::Scalar(1)/6, Plato::Scalar(1)/6, Plato::Scalar(1)/2,
//            Plato::Scalar(1)/6, Plato::Scalar(1)/2, Plato::Scalar(1)/6,
//            Plato::Scalar(1)/2, Plato::Scalar(1)/6, Plato::Scalar(1)/6
//        });
//    }

//    static inline Plato::Matrix<14,mNumSpatialDims>
//    getCubPoints()
//    {
//        return Plato::Matrix<14,mNumSpatialDims>(
//         {9.2735250310891226402323913737030605245203e-2,  9.2735250310891226402323913737030605245203e-2, 9.2735250310891226402323913737030605245203e-2,
//          7.2179424906732632079302825878890818426439e-1,  9.2735250310891226402323913737030605245203e-2, 9.2735250310891226402323913737030605245203e-2,
//          9.2735250310891226402323913737030605245203e-2,  7.2179424906732632079302825878890818426439e-1, 9.2735250310891226402323913737030605245203e-2,
//          9.2735250310891226402323913737030605245203e-2,  9.2735250310891226402323913737030605245203e-2, 7.2179424906732632079302825878890818426439e-1,
//          3.1088591926330060979734573376345783299262e-1,  3.1088591926330060979734573376345783299262e-1, 3.1088591926330060979734573376345783299262e-1,
//          6.7342242210098170607962798709626501022148e-2,  3.1088591926330060979734573376345783299262e-1, 3.1088591926330060979734573376345783299262e-1,
//          3.1088591926330060979734573376345783299262e-1,  6.7342242210098170607962798709626501022148e-2, 3.1088591926330060979734573376345783299262e-1,
//          3.1088591926330060979734573376345783299262e-1,  3.1088591926330060979734573376345783299262e-1, 6.7342242210098170607962798709626501022148e-2,
//          4.5503704125649637643858844819262773124457e-2,  4.5503704125649637643858844819262773124457e-2, 4.5449629587435036235614115518073722687554e-1,
//          4.5503704125649689297853203692977039634167e-2,  4.5449629587435031070214679630702296036583e-1, 4.5503704125649637643858844819262773124457e-2,
//          4.5449629587435031070214679630702296036583e-1,  4.5503704125649689297853203692977039634167e-2, 4.5503704125649637643858844819262773124457e-2,
//          4.5503704125649689297853203692977039634167e-2,  4.5449629587435031070214679630702296036583e-1, 4.5449629587435036235614115518073722687554e-1,
//          4.5449629587435031070214679630702296036583e-1,  4.5503704125649689297853203692977039634167e-2, 4.5449629587435036235614115518073722687554e-1,
//          4.5449629587435036235614115518073722687554e-1,  4.5449629587435036235614115518073722687554e-1, 4.5503704125649637643858844819262773124457e-2});
//    }

    DEVICE_TYPE static inline Plato::Array<mNumNodesPerCell>
    basisValues( const Plato::Array<mNumSpatialDims>& aCubPoint )
    {
        auto x=aCubPoint(0);
        auto y=aCubPoint(1);
        auto z=aCubPoint(2);
        auto x2=x*x;
        auto y2=y*y;
        auto z2=z*z;
        auto xy=x*y;
        auto yz=y*z;
        auto zx=z*x;

        Plato::Array<mNumNodesPerCell> tN;

        tN(0) = (x2+y2+z2)*2.0 - (x+y+z)*3.0 + (xy+yz+zx)*4.0 + 1.0;
        tN(1) = x2*2.0-x;
        tN(2) = y2*2.0-y;
        tN(3) = z2*2.0-z;
        tN(4) = (x-x2-xy-zx)*4.0;
        tN(5) = xy*4.0;
        tN(6) = (y-y2-xy-yz)*4.0;
        tN(7) = zx*4.0;
        tN(8) = yz*4.0;
        tN(9) = (z-z2-zx-yz)*4.0;

        return tN;
    }

    DEVICE_TYPE static inline Plato::Matrix<mNumNodesPerCell, mNumSpatialDims>
    basisGrads( const Plato::Array<mNumSpatialDims>& aCubPoint )
    {
        auto x=aCubPoint(0);
        auto y=aCubPoint(1);
        auto z=aCubPoint(2);

        Plato::Matrix<mNumNodesPerCell, mNumSpatialDims> tG;

        tG(0,0) = (x+y+z)*4.0-3.0    ; tG(0,1) = (x+y+z)*4.0-3.0    ; tG(0,2) = (x+y+z)*4.0-3.0;
        tG(1,0) = x*4.0-1.0          ; tG(1,1) = 0.0                ; tG(1,2) = 0.0;
        tG(2,0) = 0.0                ; tG(2,1) = y*4.0-1.0          ; tG(2,2) = 0.0;
        tG(3,0) = 0.0                ; tG(3,1) = 0.0                ; tG(3,2) = z*4.0-1.0;
        tG(4,0) = -(x*2.0+y+z-1)*4.0 ; tG(4,1) = -x*4.0             ; tG(4,2) = -x*4.0;
        tG(5,0) = y*4.0              ; tG(5,1) = x*4.0              ; tG(5,2) = 0.0;
        tG(6,0) = -y*4.0             ; tG(6,1) = -(x+2.0*y+z-1)*4.0 ; tG(6,2) = -y*4.0;
        tG(7,0) = z*4.0              ; tG(7,1) = 0.0                ; tG(7,2) = x*4.;
        tG(8,0) = 0.0                ; tG(8,1) = z*4.0              ; tG(8,2) = y*4.;
        tG(9,0) = -z*4.0             ; tG(9,1) = -z*4.0             ; tG(9,2) = -(x+y+z*2.0-1)*4.;

        return tG;
    }
};

} // end namespace Plato
