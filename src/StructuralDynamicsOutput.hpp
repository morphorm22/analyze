/*
 * StructuralDynamicsOutput.hpp
 *
 *  Created on: Aug 13, 2018
 */

#ifndef STRUCTURALDYNAMICSOUTPUT_HPP_
#define STRUCTURALDYNAMICSOUTPUT_HPP_

#include <memory>
#include <vector>
#include <cassert>
#include <unistd.h>

#include "PlatoMesh.hpp"

#include "SimplexStructuralDynamics.hpp"

namespace Plato
{

template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class StructuralDynamicsOutput: public Plato::SimplexStructuralDynamics<SpaceDim, NumControls>
{
private:
    static constexpr Plato::OrdinalType mSpatialDim = Plato::SimplexStructuralDynamics<SpaceDim>::mNumSpatialDims;
    static constexpr Plato::OrdinalType mNumDofsPerNode = Plato::SimplexStructuralDynamics<SpaceDim>::mNumDofsPerNode;

    Plato::MeshIO mMeshIO;

public:
    StructuralDynamicsOutput(Plato::Mesh aMesh, Plato::Scalar aRestartFreq = 0) :
            mMeshIO(nullptr)
    {
        char tTemp[FILENAME_MAX];
        auto tFilePath = getcwd(tTemp, FILENAME_MAX) ? std::string( tTemp ) : std::string("");
        assert(tFilePath.empty() == false);
        mMeshIO = Plato::MeshIOFactory::create(tFilePath, aMesh);
    }

    StructuralDynamicsOutput(Plato::Mesh aMesh, const std::string & aFilePath, Plato::Scalar aRestartFreq = 0) :
            mMeshIO(Plato::MeshIOFactory::create(aFilePath, aMesh))
    {
    }

    ~StructuralDynamicsOutput()
    {
    }

    template<typename ArrayT>
    void output(
      const ArrayT                   & tFreqArray,
      const Plato::ScalarMultiVector & aState,
            Plato::Mesh                aMesh
    )
    {
        auto tNumVertices = aMesh->nverts();
        auto tOutputNumDofs = tNumVertices * mSpatialDim;

        Plato::ScalarVector tRealDisp("RealDisp", tOutputNumDofs);
        Plato::ScalarVector tImagDisp("ImagDisp", tOutputNumDofs);

        auto tNumFrequencies = tFreqArray.size();
        for(Plato::OrdinalType tIndex = 0; tIndex < tNumFrequencies; tIndex++)
        {
            auto tMyState = Kokkos::subview(aState, tIndex, Kokkos::ALL());
            Plato::copy<mNumDofsPerNode, mSpatialDim>(/*offset=*/0, tNumVertices, tMyState, tRealDisp);
            mMeshIO->AddNodeData("RealDisp", mSpatialDim, tRealDisp);

            Plato::copy<mNumDofsPerNode, mSpatialDim>(/*offset=*/mSpatialDim, tNumVertices, tMyState, tImagDisp);
            mMeshIO->AddNodeData("ImagDisp", mSpatialDim, tImagDisp);

            auto tMyFreq = tFreqArray[tIndex];
            auto tFreqIndex = tIndex + static_cast<Plato::OrdinalType>(1);
            mMeshIO->Write(tFreqIndex, tMyFreq);
        }
    }
};
// class StructuralDynamicsOutput

} // namespace Plato

#endif /* STRUCTURALDYNAMICSOUTPUT_HPP_ */
