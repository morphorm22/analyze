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

/// @brief class for quadratic dark current density model
/// @tparam EvaluationType   automatic differentiation evaluation type, which sets scalar types
/// @tparam OutputScalarType output scalar type 
template<typename EvaluationType, 
         typename OutputScalarType = typename EvaluationType::StateScalarType>
class DarkCurrentDensityQuadratic : 
    public Plato::CurrentDensityModel<EvaluationType,OutputScalarType>
{
private:
    /// @brief topological element type
    using ElementType = typename EvaluationType::ElementType;
    /// @brief state scalar type
    using StateScalarType = typename EvaluationType::StateScalarType;
    /// @brief number of degrees of freedom per node
    static constexpr int mNumDofsPerNode  = ElementType::mNumDofsPerNode;

public:
    /// @brief default coefficient values for dark current density model quadratic model
    Plato::Scalar mCoefA  = 0.;
    Plato::Scalar mCoefB  = 1.27e-6;
    Plato::Scalar mCoefC  = 25.94253;
    Plato::Scalar mCoefM1 = 0.38886;
    Plato::Scalar mCoefB1 = 0.;
    Plato::Scalar mCoefM2 = 30.;
    Plato::Scalar mCoefB2 = 6.520373;
    Plato::Scalar mPerformanceLimit = -0.22;
    /// @brief input current density parameter list name
    std::string mCurrentDensityName = "";

public:
    /// @brief class constructor
    /// @param [in] aCurrentDensityName input current density parameter list name
    /// @param [in] aParamList          input problem parameters
    DarkCurrentDensityQuadratic(
      const std::string            & aCurrentDensityName,
      const Teuchos::ParameterList & aParamList
    ) : 
      mCurrentDensityName(aCurrentDensityName)
    {
        this->initialize(aParamList);
    }

    /// @brief class destructor
    virtual ~DarkCurrentDensityQuadratic(){}

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
        OutputScalarType tDarkCurrentDensity = 0.0;
        if( aCellElectricPotential > 0.0 )
          { tDarkCurrentDensity = mCoefA + mCoefB * exp(mCoefC * aCellElectricPotential); }
        else 
        if( (mPerformanceLimit < aCellElectricPotential) && (aCellElectricPotential < 0.0) )
          { tDarkCurrentDensity = mCoefM1 * aCellElectricPotential + mCoefB1; }
        else 
        if( aCellElectricPotential < mPerformanceLimit )
          { tDarkCurrentDensity = mCoefM2 * aCellElectricPotential + mCoefB2; }
        return tDarkCurrentDensity;
    }

    /// @fn evaluate
    /// @brief pure virtual method, evaluates current density model
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
      const Teuchos::ParameterList & aParamList
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
        this->parseCurrentDensityModelCoefficients(tCurrentDensitySublist);
    }

    /// @fn parseCurrentDensityModelCoefficients
    /// @brief parse quadratic model coefficients
    /// @param [in] aParamList input problem parameters
    void parseCurrentDensityModelCoefficients(
      Teuchos::ParameterList & aParamList
    )
    {
        mCoefA  = aParamList.get<Plato::Scalar>("a",0.);
        mCoefB  = aParamList.get<Plato::Scalar>("b",1.27e-6);
        mCoefC  = aParamList.get<Plato::Scalar>("c",25.94253);
        mCoefM1 = aParamList.get<Plato::Scalar>("m1",0.38886);
        mCoefB1 = aParamList.get<Plato::Scalar>("b1",0.);
        mCoefM2 = aParamList.get<Plato::Scalar>("m2",30.);
        mCoefB2 = aParamList.get<Plato::Scalar>("b2",6.520373);
        mPerformanceLimit = aParamList.get<Plato::Scalar>("limit",-0.22);
    }
};

}
// namespace Plato