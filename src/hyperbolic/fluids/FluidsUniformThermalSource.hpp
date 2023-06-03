/*
 * FluidsUniformThermalSource.hpp
 *
 *  Created on: June 16, 2021
 */

#pragma once

#include <Teuchos_ParameterList.hpp>

#include "BLAS2.hpp"
#include "MetaData.hpp"
#include "WorkSets.hpp"
#include "SpatialModel.hpp"
#include "UtilsTeuchos.hpp"
#include "ExpInstMacros.hpp"
#include "InterpolateFromNodal.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"

#include "hyperbolic/fluids/FluidsUtils.hpp"
#include "hyperbolic/fluids/AbstractVolumetricSource.hpp"
#include "hyperbolic/fluids/FluidsUtils.hpp"
#include "hyperbolic/fluids/SimplexFluids.hpp"
#include "hyperbolic/fluids/SimplexFluidsFadTypes.hpp"
#include "hyperbolic/fluids/EnergyConservationUtils.hpp"

namespace Plato
{

namespace Fluids
{

namespace SIMP
{

template<typename PhysicsT, typename EvaluationT>
class UniformThermalSource : public Plato::AbstractVolumetricSource<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumSpatialDims     = PhysicsT::SimplexT::mNumSpatialDims; /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell    = PhysicsT::SimplexT::mNumNodesPerCell; /*!< number of nodes per cell/element */
    static constexpr auto mNumTempDofsPerCell = PhysicsT::mNumEnergyDofsPerCell; /*!< number of degrees of freedom per cell */

    // set local ad type
    using ResultT   = typename EvaluationT::ResultScalarType; /*!< result FAD evaluation type */
    using ConfigT   = typename EvaluationT::ConfigScalarType; /*!< configuration FAD evaluation type */
    using ControlT  = typename EvaluationT::ControlScalarType; /*!< control FAD evaluation type */

    // member parameters
    Plato::Scalar mMagnitude = 0.0; /*!< thermal source magnitude */
    Plato::Scalar mPenaltyExponent = 3.0; /*!< thermal source simp penalty model exponent */
    Plato::Scalar mDimLessConstant = 1.0; /*!< dimensionless constant applied to source term */

    std::string mFuncName; /*!< scalar funciton name */
    std::vector<std::string> mElemDomains; /*!< element blocks considered for thermal source evaluation */

    // member metadata
    Plato::DataMap& mDataMap; /*!< holds output metadata */
    const Plato::SpatialDomain& mSpatialDomain; /*!< holds mesh and entity sets metadata for a domain (i.e. element block) */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< cubature integration rule */

public:
    /***************************************************************************//**
     * \brief Constructor.
     * \param [in] aFuncName function name
     * \param [in] aDomain   spatail domain database
     * \param [in] aDataMap  output database
     * \param [in] aInputs   input database
     ******************************************************************************/
    UniformThermalSource    
    (const std::string          & aFuncName,
     const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) : 
        mDataMap(aDataMap),
        mSpatialDomain(aDomain),
        mFuncName(aFuncName)
    {
        this->initialize(aInputs);
    }

    /***************************************************************************//**
     * \brief Destructor.
     ******************************************************************************/
    virtual ~UniformThermalSource(){ return; }
    
    /***************************************************************************//**
     * \brief Return function type.
     * \return function type
     ******************************************************************************/
    std::string type() const override
    {
        return "uniform";
    }
    
    /***************************************************************************//**
     * \brief Return function name.
     * \return function name
     ******************************************************************************/
    std::string name() const override
    {
        return mFuncName;
    }

    /***************************************************************************//**
     * \brief Evaluate thermal source integral.
     * \param [in]  aWorkSets   workset database
     * \param [out] aResultWS   output/result workset
     * \param [in]  aMultiplier scalar multiplier (default = 1.0)
     ******************************************************************************/
    void evaluate
    (const Plato::WorkSets & aWorkSets, 
     Plato::ScalarMultiVectorT<ResultT> & aResultWS,
     Plato::Scalar aMultiplier = 1.0) 
     const override
    {
        auto tMyBlockName = mSpatialDomain.getElementBlockName();
        auto tEvaluateDomain = std::find(mElemDomains.begin(), mElemDomains.end(), tMyBlockName) != mElemDomains.end();
        if( tEvaluateDomain )
        {
            auto tNumCells = mSpatialDomain.numCells();
            if (tNumCells != static_cast<Plato::OrdinalType>(aResultWS.extent(0)) )
            {
                ANALYZE_THROWERR(std::string("Number of elements mismatch. Spatial domain and output/result workset ") 
                    + "cell number does not match. " + "Spatial domain has '" + std::to_string(tNumCells) 
                    + "' cells/elements and output workset has '" + std::to_string(aResultWS.extent(0)) + "' cells/elements.")
            }

            // set local functors
            Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;

            // set constant thermal source
            Plato::ScalarVectorT<Plato::Scalar> tThermalSource("thermal source", tNumCells);
            Plato::blas1::fill(mMagnitude, tThermalSource);

            // set local arrays
            Plato::ScalarVectorT<ConfigT> tCellVolume("cell weight", tNumCells);
            Plato::ScalarArray3DT<ConfigT> tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);

            // set input state worksets
            auto tConfigWS = Plato::unpack<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
            auto tControlWS = Plato::unpack<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("control"));

            // transfer member data to device
            auto tDimLessConstant = mDimLessConstant;
            auto tPenaltyExponent = mPenaltyExponent;

            auto tCubWeight = mCubatureRule.getCubWeight();
            auto tBasisFunctions = mCubatureRule.getBasisFunctions();
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), KOKKOS_LAMBDA(const Plato::OrdinalType &aCellOrdinal)
            {
                tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
                tCellVolume(aCellOrdinal) = tCellVolume(aCellOrdinal) * tCubWeight;

                auto tUnpenalizedDimLessConstant = aMultiplier * tDimLessConstant;
                ControlT tPenalizedDimLessConstant = Plato::Fluids::penalize_heat_source_constant<mNumNodesPerCell>
                    (aCellOrdinal, tUnpenalizedDimLessConstant, tPenaltyExponent, tControlWS);
                Plato::Fluids::integrate_scalar_field<mNumTempDofsPerCell>
                    (aCellOrdinal, tBasisFunctions, tCellVolume, tThermalSource, aResultWS, -tPenalizedDimLessConstant);
            },"intergate thermal source term");
        }
    }

private:
    /***************************************************************************//**
     * \brief Initialize thermal source.
     * \param [in] aInputs  input database
     ******************************************************************************/
    void initialize(Teuchos::ParameterList& aInputs)
    {
        if( aInputs.isSublist("Thermal Sources") )
        {
            auto tMaterialName = mSpatialDomain.getMaterialName();
            mDimLessConstant = Plato::Fluids::compute_thermal_source_dimensionless_constant(tMaterialName, aInputs);

            auto tThermalSourceParamList = aInputs.sublist("Thermal Sources");
            mMagnitude = Plato::teuchos::parse_parameter<Plato::Scalar>("Value", mFuncName, tThermalSourceParamList);
            
            this->parseDomains(aInputs);
            this->parseMaterialPenaltyModel(aInputs);
        }
    }

    /***************************************************************************//**
     * \brief Initialize topology penalization model parameters.
     * \param [in] aInputs  input database
     ******************************************************************************/
    void parseMaterialPenaltyModel(Teuchos::ParameterList& aInputs)
    {
        auto tMaterialName = mSpatialDomain.getMaterialName();
        if(Plato::Fluids::is_material_property_defined("Source Term Penalty Exponent", tMaterialName, aInputs))
        {
            mPenaltyExponent = Plato::Fluids::get_material_property<Plato::Scalar>("Source Term Penalty Exponent", tMaterialName, aInputs);
            Plato::is_positive_finite_number(mPenaltyExponent, "Source Term Penalty Exponent");
        }
    }

    /***************************************************************************//**
     * \brief Parse domains where thermal source will be evaluated.
     * \param [in] aInputs input database
     ******************************************************************************/
    void parseDomains(Teuchos::ParameterList& aInputs)
    {
        auto tParamList = aInputs.sublist("Thermal Source");
        mElemDomains = Plato::teuchos::parse_array<std::string>("Domains", tParamList);
        if( mElemDomains.empty() )
        {
            // default: use all the element blocks for the thermal source evaluation
            auto tMyBlockName = mSpatialDomain.getElementBlockName();
            mElemDomains.push_back(tMyBlockName);
        }
    }
};
// class UniformThermalSource

}
// namespace SIMP

template<typename PhysicsT, typename EvaluationT>
class UniformThermalSource : public Plato::AbstractVolumetricSource<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumSpatialDims  = PhysicsT::SimplexT::mNumSpatialDims; /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell = PhysicsT::SimplexT::mNumNodesPerCell; /*!< number of nodes per cell/element */
    static constexpr auto mNumTempDofsPerCell = PhysicsT::mNumEnergyDofsPerCell; /*!< number of degrees of freedom per cell */

    // set local ad type
    using ResultT   = typename EvaluationT::ResultScalarType; /*!< result FAD evaluation type */
    using ConfigT   = typename EvaluationT::ConfigScalarType; /*!< configuration FAD evaluation type */

    // member parameters
    Plato::Scalar mMagnitude = 0.0; /*!< thermal source magnitude */
    Plato::Scalar mDimLessConstant = 1.0; /*!< dimensionless constant applied to source term */

    std::string mFuncName; /*!< scalar funciton name */
    std::vector<std::string> mElemDomains; /*!< element blocks considered for thermal source evaluation */

    // member metadata
    Plato::DataMap& mDataMap; /*!< holds output metadata */
    const Plato::SpatialDomain& mSpatialDomain; /*!< holds mesh and entity sets metadata for a domain (i.e. element block) */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< cubature integration rule */

public:
    /***************************************************************************//**
     * \brief Constructor.
     * \param [in] aFuncName function name
     * \param [in] aDomain   spatail domain database
     * \param [in] aDataMap  output database
     * \param [in] aInputs   input database
     ******************************************************************************/
    UniformThermalSource    
    (const std::string          & aFuncName,
     const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) : 
        mDataMap(aDataMap),
        mSpatialDomain(aDomain),
        mFuncName(aFuncName)
    {
        this->initialize(aInputs);
    }
    
    /***************************************************************************//**
     * \brief Destructor
     ******************************************************************************/
    virtual ~UniformThermalSource(){ return; }
    
    /***************************************************************************//**
     * \brief Return function type.
     * \return function type
     ******************************************************************************/
    std::string type() const override
    {
        return "uniform";
    }

    /***************************************************************************//**
     * \brief Return function name.
     * \return function name
     ******************************************************************************/
    std::string name() const override
    {
        return mFuncName;
    }

    /***************************************************************************//**
     * \brief Evaluate thermal source integral.
     * \param [in]  aWorkSets   workset database
     * \param [out] aResultWS   output/result workset
     * \param [in]  aMultiplier scalar multiplier (default = 1.0)
     ******************************************************************************/
    void evaluate
    (const Plato::WorkSets & aWorkSets, 
     Plato::ScalarMultiVectorT<ResultT> & aResultWS,
     Plato::Scalar aMultiplier = 1.0) 
     const override
    {
        auto tMyBlockName = mSpatialDomain.getElementBlockName();
        auto tEvaluateDomain = std::find(mElemDomains.begin(), mElemDomains.end(), tMyBlockName) != mElemDomains.end();
        if( tEvaluateDomain )
        {
            auto tNumCells = mSpatialDomain.numCells();
            if (tNumCells != static_cast<Plato::OrdinalType>(aResultWS.extent(0)) )
            {
                ANALYZE_THROWERR(std::string("Number of elements mismatch. Spatial domain and output/result workset ") 
                    + "cell number does not match. " + "Spatial domain has '" + std::to_string(tNumCells) 
                    + "' cells/elements and output workset has '" + std::to_string(aResultWS.extent(0)) + "' cells/elements.")
            }

            // set local functors
            Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;

            // set constant thermal source
            Plato::ScalarVectorT<Plato::Scalar> tThermalSource("thermal source", tNumCells);
            Plato::blas1::fill(mMagnitude, tThermalSource);

            // set local arrays
            Plato::ScalarVectorT<ConfigT> tCellVolume("cell weight", tNumCells);
            Plato::ScalarArray3DT<ConfigT> tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);

            // set input state worksets
            auto tConfigWS = Plato::unpack<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));

            // transfer member data to device
            auto tDimLessConstant = mDimLessConstant;

            auto tCubWeight = mCubatureRule.getCubWeight();
            auto tBasisFunctions = mCubatureRule.getBasisFunctions();
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), KOKKOS_LAMBDA(const Plato::OrdinalType &aCellOrdinal)
            {
                tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
                tCellVolume(aCellOrdinal) = tCellVolume(aCellOrdinal) * tCubWeight;

                auto tScalarConstant = aMultiplier * tDimLessConstant;
                Plato::Fluids::integrate_scalar_field<mNumTempDofsPerCell>
                    (aCellOrdinal, tBasisFunctions, tCellVolume, tThermalSource, aResultWS, -tScalarConstant);
            },"intergate thermal source term");
        }
    }

private:
    /***************************************************************************//**
     * \brief Initialize thermal source.
     * \param [in] aInputs  input database
     ******************************************************************************/
    void initialize(Teuchos::ParameterList& aInputs)
    {
        if( aInputs.isSublist("Thermal Source") )
        {
            auto tMaterialName = mSpatialDomain.getMaterialName();
            mDimLessConstant = Plato::Fluids::compute_thermal_source_dimensionless_constant(tMaterialName, aInputs);

            auto tThermalSourceParamList = aInputs.sublist("Thermal Source");
            mMagnitude = Plato::teuchos::parse_parameter<Plato::Scalar>("Value", mFuncName, tThermalSourceParamList);

            this->parseDomains(aInputs);
        }
    }

    /***************************************************************************//**
     * \brief Parse domains where thermal source will be evaluated.
     * \param [in] aInputs input database
     ******************************************************************************/
    void parseDomains(Teuchos::ParameterList& aInputs)
    {
        auto tParamList = aInputs.sublist("Thermal Source");
        mElemDomains = Plato::teuchos::parse_array<std::string>("Domains", tParamList);
        if( mElemDomains.empty() )
        {
            // default: use all the element blocks for the thermal source evaluation
            auto tMyBlockName = mSpatialDomain.getElementBlockName();
            mElemDomains.push_back(tMyBlockName);
        }
    }
};
// class UniformThermalSource

}
// namespace Fluids

}
// namespace Plato

#include "hyperbolic/IncompressibleFluids.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::UniformThermalSource, Plato::IncompressibleFluids, Plato::SimplexFluids, 1, 1)
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::SIMP::UniformThermalSource, Plato::IncompressibleFluids, Plato::SimplexFluids, 1, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::UniformThermalSource, Plato::IncompressibleFluids, Plato::SimplexFluids, 2, 1)
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::SIMP::UniformThermalSource, Plato::IncompressibleFluids, Plato::SimplexFluids, 2, 1)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::UniformThermalSource, Plato::IncompressibleFluids, Plato::SimplexFluids, 3, 1)
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::SIMP::UniformThermalSource, Plato::IncompressibleFluids, Plato::SimplexFluids, 3, 1)
#endif
