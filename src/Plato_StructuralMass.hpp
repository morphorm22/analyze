/*
 * Plato_StructuralMass.hpp
 *
 *  Created on: Apr 17, 2019
 */

#pragma once

#include "BLAS1.hpp"
#include "Simplex.hpp"
#include "ImplicitFunctors.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Interface to compute the structural mass
**********************************************************************************/
template<Plato::OrdinalType SpaceDim>
class StructuralMass : public Plato::Simplex<SpaceDim>
{
private:
    static constexpr auto mNumNodesPerCell = Plato::Simplex<SpaceDim>::mNumNodesPerCell; /*!< number of nodes per element/cell */

    Plato::Scalar mMaterialDensity; /*!< material density (note: constant for all elements/cells) */

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aMaterialDensity material density (note: constant for all elements/cells)
    **********************************************************************************/
    explicit StructuralMass(const Plato::Scalar & aMaterialDensity) :
            mMaterialDensity(aMaterialDensity)
    {
    }

    /******************************************************************************//**
     * \brief Destructor
    **********************************************************************************/
    ~StructuralMass()
    {
    }

    /******************************************************************************//**
     * \brief Compute the total structural mass
     * \param [in] aNumCells number of elements/cells
     * \param [in] aControl design variables used to denote material or void
     * \param [in] aConfig coordinates
     * \param [out] aOutput total structural mass
    **********************************************************************************/
    template<typename OutputType, typename ControlType, typename ConfigType>
    inline void operator()(const Plato::OrdinalType aNumCells,
                           const Plato::ScalarMultiVectorT<ControlType> aControl,
                           const Plato::ScalarArray3DT<ConfigType> aConfig,
                           OutputType & aOutput) const
    {
        Plato::ComputeCellVolume<SpaceDim> tComputeCellVolume;
        Plato::LinearTetCubRuleDegreeOne<SpaceDim> tCubatureRule;

        auto tMaterialDensity = mMaterialDensity;
        Plato::ScalarVectorT<OutputType> tTotalMass("total mass", aNumCells);

        auto tCubWeight = tCubatureRule.getCubWeight();
        auto tBasisFunc = tCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), KOKKOS_LAMBDA(const Plato::OrdinalType & aCellOrdinal)
        {
            ConfigType tCellVolume = 0;
            tComputeCellVolume(aCellOrdinal, aConfig, tCellVolume);
            ControlType tCellMass = Plato::cell_mass<mNumNodesPerCell>(aCellOrdinal, tBasisFunc, aControl);
            tTotalMass(aCellOrdinal) = tCellMass * tMaterialDensity * tCellVolume * tCubWeight;
        },"Compute Structural Mass");

        aOutput = 0;
        Plato::blas1::local_sum(tTotalMass, aOutput);
    }
};
// class StructuralMass

} // namespace Plato

#ifdef PLATOANALYZE_1D
extern template class Plato::StructuralMass<1>;
#endif

#ifdef PLATOANALYZE_2D
extern template class Plato::StructuralMass<2>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::StructuralMass<3>;
#endif
