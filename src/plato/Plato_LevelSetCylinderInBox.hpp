/*
 //@HEADER
 // *************************************************************************
 //   Plato Engine v.1.0: Copyright 2018, National Technology & Engineering
 //                    Solutions of Sandia, LLC (NTESS).
 //
 // Under the terms of Contract DE-NA0003525 with NTESS,
 // the U.S. Government retains certain rights in this software.
 //
 // Redistribution and use in source and binary forms, with or without
 // modification, are permitted provided that the following conditions are
 // met:
 //
 // 1. Redistributions of source code must retain the above copyright
 // notice, this list of conditions and the following disclaimer.
 //
 // 2. Redistributions in binary form must reproduce the above copyright
 // notice, this list of conditions and the following disclaimer in the
 // documentation and/or other materials provided with the distribution.
 //
 // 3. Neither the name of the Sandia Corporation nor the names of the
 // contributors may be used to endorse or promote products derived from
 // this software without specific prior written permission.
 //
 // THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
 // EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 // PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
 // CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 // EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 // PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 // PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 // LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 // NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 // SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 //
 // Questions? Contact the Plato team (plato3D-help@sandia.gov)
 //
 // *************************************************************************
 //@HEADER
 */

/*
 * Plato_LevelSetCylinderInBox.hpp
 *
 *  Created on: Aug 29, 2018
 */

#pragma once

#include "HamiltonJacobi.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_file.hpp"

#define _USE_MATH_DEFINES

#include <map>
#include <string>
#include <math.h>
#include <vector>
#include <memory>
#include <cassert>
#include <cstddef>
#include <cstdlib>

#include "Plato_GeometryModel.hpp"

namespace Plato
{

struct LevelSetInitialCondition
{
    LevelSetInitialCondition(Plato::Scalar radius, Plato::Scalar length) :
            Nlobes(5),
            Rmid(radius),
            Rdelta(0),
            xCenter(0.5 * length),
            yCenter(0.5 * length)
    {
    }
    const int Nlobes;
    const Plato::Scalar Rmid;
    const Plato::Scalar Rdelta;
    const Plato::Scalar xCenter;
    const Plato::Scalar yCenter;

    Plato::Scalar operator()(Plato::Scalar x, Plato::Scalar y, Plato::Scalar z) const
    {
        const Plato::Scalar r = std::sqrt((x - xCenter) * (x - xCenter) + (y - yCenter) * (y - yCenter));
        const Plato::Scalar theta = atan2(y - yCenter, x - xCenter);
        const Plato::Scalar rSurf = Rmid + Rdelta * sin(Nlobes * theta);
        return r - rSurf;
    }
};

/******************************************************************************//**
 * @brief Cylinder geometry model class
 **********************************************************************************/
template<typename ScalarType = double>
class LevelSetCylinderInBox : public Plato::GeometryModel<ScalarType>
{
public:
    static constexpr int mSpatialDim = 3;

    /******************************************************************************//**
     * @brief Default constructor
     **********************************************************************************/
    explicit LevelSetCylinderInBox(MPI_Comm aComm = MPI_COMM_WORLD) :
            mComm(aComm)
    {
    }

    /******************************************************************************//**
     * @brief Destructor
     **********************************************************************************/
    virtual ~LevelSetCylinderInBox()
    {
    }

    /******************************************************************************//**
     * @brief Return cylinder's radius
     * @return radius
     **********************************************************************************/
    ScalarType radius() const
    {
        return (mRadius);
    }

    /******************************************************************************//**
     * @brief Return cylinder's length
     * @return length
     **********************************************************************************/
    ScalarType length() const
    {
        return (mLength);
    }

    /******************************************************************************//**
     * @brief compute the area of the side of a cylinder.
     **********************************************************************************/
    ScalarType area()
    {
        const ScalarType tArea = level_set_area(mMesh, mHamiltonJacobiFields, mInterfaceWidth);
        return (tArea);
    }

    /******************************************************************************//**
     * @brief Compute the reference rate that gas mass is begin produced
     * @return mass production rate
     **********************************************************************************/
    ScalarType referencMassProductionRate()
    {
        const ScalarType tRefMassProdRate =
                mPropellantDensity * mRefBurnRate * level_set_volume_rate_of_change(mMesh, mHamiltonJacobiFields, mInterfaceWidth);
        return tRefMassProdRate;
    }

    /******************************************************************************//**
     * @brief compute the gradient of a cylinder with respect to parameters that define geometry.
     * @param aOutput gradient with respect to the parameters that defined a geometry
     **********************************************************************************/
    void gradient(std::vector<ScalarType>& aOutput)
    {
        return;
    }

    /******************************************************************************//**
     * @brief Evolve geometry in time
     * @param [in] aDeltaTime time step
     * @param [in] aBurnRateMultiplier actual burn rate divided by the reference burn rate
     **********************************************************************************/
    virtual void evolveGeometry(const ScalarType aDeltaTime, const ScalarType aBurnRateMultiplier)
    {
        this->updateLevelSetCylinder(aDeltaTime, aBurnRateMultiplier);
    }

    /******************************************************************************//**
     * @brief Update immersed geometry
     * @param [in] aParam optimization parameters
     **********************************************************************************/
    void updateGeometry(const Plato::ProblemParams & aParam)
    {
        assert(aParam.mGeometry.size() == static_cast<size_t>(2));
        mRadius = aParam.mGeometry[0];
        mLength = aParam.mGeometry[1];
        assert(aParam.mRefBurnRate.size() == static_cast<size_t>(1));
        mRefBurnRate = aParam.mRefBurnRate[0];
        mPropellantDensity = aParam.mPropellantDensity;

        this->initializeLevelSetCylinder();
    }

    /******************************************************************************//**
     * @brief Initialize level set cylinder
     * @param [in] aParam parameters associated with the geometry and fields
     **********************************************************************************/
    void initialize(const Plato::ProblemParams & aParam)
    {
        this->updateGeometry(aParam);
    }

private:
    /******************************************************************************//**
     * @brief Update immersed cylinder
     * @param [in] aParam optimization parameters
     **********************************************************************************/
    void updateLevelSetCylinder(const ScalarType aDeltaTime, const ScalarType aBurnRateMultiplier)
    {
        const Plato::Scalar tDeltaPseudoTime = aBurnRateMultiplier * mRefBurnRate * aDeltaTime;
        offset_level_set(mMesh, mHamiltonJacobiFields, -tDeltaPseudoTime);
        mTime += aDeltaTime;
        ++mStep;
    }

    /******************************************************************************//**
     * @brief Initialize immersed cylinder
     **********************************************************************************/
    void initializeLevelSetCylinder()
    {
        if(mLength != mMeshLength)
        {
            build_mesh(mLength);
            mWriter = Omega_h::vtk::Writer("LevelSetCylinderInBox", &mMesh, mSpatialDim);
            write_mesh(mWriter, mMesh, mHamiltonJacobiFields, mTime);
        }

        LevelSetInitialCondition tInitialCondition(mRadius, mLength);
        initialize_level_set(mMesh, mHamiltonJacobiFields, tInitialCondition);

        const Plato::Scalar tRelativeSpeed = 1.0;
        const Plato::Scalar tMaxSpeed = initialize_constant_speed(mMesh, mHamiltonJacobiFields, tRelativeSpeed);
        const Plato::Scalar tDeltaX = mesh_minimum_length_scale<mSpatialDim>(mMesh);
        const Plato::Scalar tDeltaTau = static_cast<Plato::Scalar>(0.2) * tDeltaX / tMaxSpeed; // Units of time
        mInterfaceWidth = static_cast<Plato::Scalar>(1.5) * tDeltaX / tMaxSpeed; // Should have same units as level set

        compute_arrival_time(mMesh, mHamiltonJacobiFields, mInterfaceWidth, tDeltaTau);
    }

    /******************************************************************************//**
     * @brief Build bounding box and fields on computational mesh
     **********************************************************************************/
    void build_mesh(const ScalarType aMeshLength)
    {
        mMeshLength = aMeshLength;
        auto tLibOmegaH = std::make_shared < Omega_h::Library > (nullptr, nullptr, mComm);
        const Plato::Scalar tLengthX = mMeshLength;
        const Plato::Scalar tLengthY = mMeshLength;
        const Plato::Scalar tLengthZ = mMeshLength;
        const size_t tNumCellsPerSide = 64;
        const size_t tNumCellX = tNumCellsPerSide;
        const size_t tNumCellY = tNumCellsPerSide;
        const size_t tNumCellZ = tNumCellsPerSide;
        mMesh = Omega_h::build_box(tLibOmegaH->world(),
                                   OMEGA_H_SIMPLEX,
                                   tLengthX,
                                   tLengthY,
                                   tLengthZ,
                                   tNumCellX,
                                   tNumCellY,
                                   tNumCellZ);
        declare_fields(mMesh, mHamiltonJacobiFields);
    }

    /******************************************************************************//**
     * @brief Write level set field on mesh every N number of time steps
     **********************************************************************************/
    void outputLevelSetField()
    {
        const size_t tPrintInterval = 1000; // How often do you want to output mesh?
        if(mStep % tPrintInterval == 0)
        {
            write_mesh(mWriter, mMesh, mHamiltonJacobiFields, mTime);
        }
    }

private:
    MPI_Comm mComm;
    ProblemFields<mSpatialDim> mHamiltonJacobiFields;
    Omega_h::Mesh mMesh;
    Omega_h::vtk::Writer mWriter;
    ScalarType mRadius = 0.0;
    ScalarType mLength = 0.0;
    ScalarType mMeshLength = 0.0;
    ScalarType mPropellantDensity = 0.0;
    ScalarType mRefBurnRate = 0.0;
    ScalarType mInterfaceWidth = 0.0;
    ScalarType mTime = 0.0;
    size_t mStep = 0;
};
// class LevelSetCylinderInBox

}// namespace Plato
