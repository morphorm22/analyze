/*
 * DarkCurrentDensityQuadratic.hpp
 *
 *  Created on: May 24, 2023
 */

#pragma once

/// @include trilinos includes
#include <Teuchos_ParameterList.hpp>

/// @include analyze includes
#include "AnalyzeMacros.hpp"
#include "InterpolateFromNodal.hpp"
#include "elliptic/electrical/CurrentDensityModel.hpp"

namespace Plato
{

/// @brief class for constant ligh-generated current density model of the form
///   \f$ J = \alpha\beta \f$, where \f$\alpha\f$ is the generation rate and 
///   \f$\beta\f$ is the solar illumination power.
/// @tparam EvaluationType   automatic differentiation evaluation type, which sets scalar types
/// @tparam OutputScalarType output scalar type 
template<typename EvaluationType, 
         typename OutputScalarType = Plato::Scalar>
class LightGeneratedCurrentDensityConstant : 
    public Plato::CurrentDensityModel<EvaluationType,OutputScalarType>
{
private:
    /// @brief topological element type
    using ElementType = typename EvaluationType::ElementType;
    /// @brief state scalar type
    using StateScalarType = typename EvaluationType::StateScalarType;
    /// @brief number of degrees of freedom per vertex/node
    static constexpr int mNumDofsPerNode  = ElementType::mNumDofsPerNode;

public:
    /// @brief name of light-generated current density input parameter list 
    std::string mCurrentDensityName = "";
    /// @brief generation rate coefficient
    Plato::Scalar mGenerationRate = -0.40914;
    /// @brief solar illumination power coefficient 
    Plato::Scalar mIlluminationPower = 1000.0;

public:
    /// @brief class constructor
    /// @param [in] aCurrentDensityName input current density parameter list name
    /// @param [in] aParamList          input problem parameters
    LightGeneratedCurrentDensityConstant(
      const std::string            & aCurrentDensityName,
      const Teuchos::ParameterList & aParamList
    ) : 
      mCurrentDensityName(aCurrentDensityName)
    {
      this->initialize(aParamList);
    }

    /// @brief class destructor
    virtual ~LightGeneratedCurrentDensityConstant(){}

    /// @fn evaluate
    /// @brief evaluate cell current density model
    /// @param [in] aCellElectricPotential cell electric potential
    /// @return scalar value 
    KOKKOS_INLINE_FUNCTION
    OutputScalarType 
    evaluate(
        const StateScalarType & aCellElectricPotential
    ) const
    {
      Plato::Scalar tOutput = mGenerationRate * mIlluminationPower;
      return ( tOutput );
    }

    /// @fn evaluate
    /// @brief implements pure virtual method, evaluates current density model
    /// @param [in] aState  2D state workset
    /// @param [in] aResult 2D output workset
    void evaluate(
      const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
      const Plato::ScalarMultiVectorT <OutputScalarType>  & aResult
    ) const
    {
        // integration rule
        auto tCubPoints  = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints  = tCubWeights.size();
        
        Plato::InterpolateFromNodal<ElementType,mNumDofsPerNode> tInterpolateFromNodal;

        // evaluate light-generated current density
        Plato::OrdinalType tNumCells = aState.extent(0);
        Kokkos::parallel_for("light-generated current density", 
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
          KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            auto tCubPoint = tCubPoints(iGpOrdinal);
            auto tBasisValues = ElementType::basisValues(tCubPoint);
            // evaluate light-generated current density
            StateScalarType tCellElectricPotential = tInterpolateFromNodal(iCellOrdinal,tBasisValues,aState);
            aResult(iCellOrdinal,iGpOrdinal) = this->evaluate(tCellElectricPotential);
        });
    }

private:
    /// @fn initialize
    /// @brief initialize current density model
    /// @param [in] aParamList input problem parameters
    void 
    initialize(
      const Teuchos::ParameterList &aParamList
    )
    {
        if( !aParamList.isSublist("Source Terms") ){
          auto tMsg = std::string("Parameter is not valid. Argument ('Source Terms') is not a parameter list");
          ANALYZE_THROWERR(tMsg)
        }
        auto tSourceTermsSublist = aParamList.sublist("Source Terms");

        if( !tSourceTermsSublist.isSublist(mCurrentDensityName) ){
          auto tMsg = std::string("Parameter is not valid. Argument ('") + mCurrentDensityName 
            + "') is not a parameter list";
          ANALYZE_THROWERR(tMsg)
        }
        auto tCurrentDensitySublist = tSourceTermsSublist.sublist(mCurrentDensityName);
        mGenerationRate = tCurrentDensitySublist.get<Plato::Scalar>("Generation Rate",-0.40914);
        mIlluminationPower = tCurrentDensitySublist.get<Plato::Scalar>("Illumination Power",1000.);
    }
};

}
// namespace Plato