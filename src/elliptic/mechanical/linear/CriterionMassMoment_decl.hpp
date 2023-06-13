#pragma once

#include "base/CriterionBase.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Mass moment class
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
**********************************************************************************/
template<typename EvaluationType>
class CriterionMassMoment : public Plato::CriterionBase
/******************************************************************************/
{
  private:
    using ElementType = typename EvaluationType::ElementType;

    static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;
    static constexpr auto mNumDofsPerNode  = ElementType::mNumDofsPerNode;
    static constexpr auto mNumDofsPerCell  = ElementType::mNumDofsPerCell;
    static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;

    using FunctionBaseType = typename Plato::CriterionBase;
    using FunctionBaseType::mSpatialDomain;
    using FunctionBaseType::mDataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    Plato::Scalar mMassDensity = 1.0;         /*!< material density */
    Plato::Scalar mTotalStructuralMass = 1.0; /*!< total structural mass, used for criterion normalization purposes*/

    std::string mCalculationType = "";  /*!< calculation type = Mass, CGx, CGy, CGz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz */
    bool mNormalizeCriterion = false;   /*!< normalize criterion */

  public:
    /******************************************************************************//**
     * \brief Primary constructor
     * \param [in] aSpatialDomain Plato Analyze spatial domain 
     * \param [in] aDataMap Plato Analyze data map
     * \param [in] aInputParams input parameters database
     * \param [in] aFuncName function name
     **********************************************************************************/
    CriterionMassMoment(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aInputParams,
              std::string             aFuncName = "CriterionMassMoment"
    );

    /******************************************************************************//**
     * \brief Unit testing constructor
     * \param [in] aDataMap PLATO Engine and Analyze data map
     **********************************************************************************/
    CriterionMassMoment(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap& aDataMap
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

    /// @fn isLinear
    /// @brief returns true if criterion is linear
    /// @return boolean
    bool 
    isLinear() 
    const;

    /// @brief evaluate mass moment criterion
    /// @param [in,out] aWorkSets function domain and range workset database
    /// @param [in]     aCycle    scalar 
    void
    evaluateConditional(
      const Plato::WorkSets & aWorkSets,
      const Plato::Scalar   & aCycle
    ) const;

    /******************************************************************************//**
     * \brief Compute structural mass
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [out] aResult 1D container of cell criterion values
     * \param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    void
    computeStructuralMass(
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const;

    /******************************************************************************//**
     * \brief Compute first mass moment
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [in] aComponent vector component (e.g. x = 0 , y = 1, z = 2)
     * \param [out] aResult 1D container of cell criterion values
     * \param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    void
    computeFirstMoment(
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::OrdinalType  aComponent,
              Plato::Scalar       aTimeStep = 0.0
    ) const;

    /******************************************************************************//**
     * \brief Compute second mass moment
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [in] aComponent1 vector component (e.g. x = 0 , y = 1, z = 2)
     * \param [in] aComponent2 vector component (e.g. x = 0 , y = 1, z = 2)
     * \param [out] aResult 1D container of cell criterion values
     * \param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    void
    computeSecondMoment(
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::OrdinalType aComponent1,
              Plato::OrdinalType aComponent2,
              Plato::Scalar      aTimeStep = 0.0
    ) const;

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

    /******************************************************************************//**
     * \brief Compute total structural mass
     **********************************************************************************/
    void computeTotalStructuralMass();

  private:

    /******************************************************************************//**
     * \brief Initialize member data
     * \param [in] aSpatialDomain spatial domain; e.g., element block, information
     * \param [in] aInputParams   input parameters database
     **********************************************************************************/
    void initialize(
      const Plato::SpatialDomain   & aSpatialDomain,
            Teuchos::ParameterList & aInputParams
    );

    /******************************************************************************//**
     * \brief Parse material density
     * \param [in] aSpatialDomain spatial domain; e.g., element block, information
     * \param [in] aInputParams   input parameters database
     **********************************************************************************/
    void parseMaterialDensity(
      const Plato::SpatialDomain   & aSpatialDomain,
            Teuchos::ParameterList & aInputParams
    );

    void
    parseNormalizeCriterion(
      const Plato::SpatialDomain   & aSpatialDomain,
            Teuchos::ParameterList & aInputParams
    );
};
// class CriterionMassMoment

} // namespace Elliptic

} // namespace Plato
