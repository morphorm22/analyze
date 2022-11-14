#pragma once

#include "PlatoStaticsTypes.hpp"
#include "geometric/AbstractScalarFunction.hpp"

namespace Plato
{

namespace Geometric
{

/******************************************************************************//**
 * \brief Mass moment class
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
**********************************************************************************/
template<typename EvaluationType>
class MassMoment :
    public EvaluationType::ElementType,
    public Plato::Geometric::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumNodesPerCell;
    using ElementType::mNumSpatialDims;
    
    using FunctionBaseType = typename Plato::Geometric::AbstractScalarFunction<EvaluationType>;
    using FunctionBaseType::mSpatialDomain;
    using FunctionBaseType::mDataMap;

    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    Plato::Scalar mCellMaterialDensity; /*!< material density */

    /*!< calculation type = Mass, CGx, CGy, CGz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz */
    std::string mCalculationType;

  public:
    /******************************************************************************//**
     * \brief Primary constructor
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap Plato Analyze data map
     * \param [in] aInputParams input parameters database
     **********************************************************************************/
    MassMoment(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aInputParams
    );

    /******************************************************************************//**
     * \brief Unit testing constructor
     * \param [in] aMesh mesh database
     * \param [in] aDataMap Plato Engine and Analyze data map
     **********************************************************************************/
    MassMoment(
        const Plato::SpatialDomain & aSpatialDomain,
              Plato::DataMap       & aDataMap
    );

    /******************************************************************************//**
     * \brief set material density
     * \param [in] aMaterialDensity material density
     **********************************************************************************/
    void setMaterialDensity(const Plato::Scalar aMaterialDensity);

    /******************************************************************************//**
     * \brief set calculation type
     * \param [in] aCalculationType calculation type string
     **********************************************************************************/
    void setCalculationType(const std::string & aCalculationType);

    /******************************************************************************//**
     * \brief Evaluate mass moment function
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [out] aResult 1D container of cell criterion values
    **********************************************************************************/
    void
    evaluate_conditional(
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult
    ) const override;

    /******************************************************************************//**
     * \brief Compute structural mass
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [out] aResult 1D container of cell criterion values
    **********************************************************************************/
    void
    computeStructuralMass(
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult
    ) const;

    /******************************************************************************//**
     * \brief Compute first mass moment
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [in] aComponent vector component (e.g. x = 0 , y = 1, z = 2)
     * \param [out] aResult 1D container of cell criterion values
    **********************************************************************************/
    void
    computeFirstMoment(
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::OrdinalType aComponent
    ) const ;

    /******************************************************************************//**
     * \brief Compute second mass moment
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [in] aComponent1 vector component (e.g. x = 0 , y = 1, z = 2)
     * \param [in] aComponent2 vector component (e.g. x = 0 , y = 1, z = 2)
     * \param [out] aResult 1D container of cell criterion values
    **********************************************************************************/
    void
    computeSecondMoment(
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::OrdinalType aComponent1,
              Plato::OrdinalType aComponent2
    ) const ;

    /******************************************************************************//**
     * \brief Map quadrature points to physical domain
     * \param [in] aRefPoint incoming quadrature points
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [out] aMappedPoints points mapped to physical domain
    **********************************************************************************/
    void
    mapQuadraturePoints(
        const Plato::ScalarArray3DT <ConfigScalarType> & aConfig,
              Plato::ScalarArray3DT <ConfigScalarType> & aMappedPoints
    ) const;
};
// class MassMoment

} // namespace Geometric

} // namespace Plato
