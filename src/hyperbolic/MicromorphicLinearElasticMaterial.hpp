#pragma once

#include <Omega_h_matrix.hpp>
#include <Teuchos_ParameterList.hpp>

#include "AnalyzeMacros.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Base class for micromorphic Linear Elastic material models
 *
 * \tparam SpatialDim spatial dimensions, options 1D, 2D, and 3D
 *
**********************************************************************************/
template<Plato::OrdinalType SpatialDim>
class MicromorphicLinearElasticMaterial
{
protected:
    static constexpr auto mNumVoigtTerms   = (SpatialDim == 3) ? 6 :
                                             ((SpatialDim == 2) ? 3 :
                                            (((SpatialDim == 1) ? 1 : 0)));
    static constexpr auto mNumSkwTerms     = (SpatialDim == 3) ? 3 :
                                             ((SpatialDim == 2) ? 1 :
                                            (((SpatialDim == 1) ? 1 : 0)));


    static_assert(mNumVoigtTerms, "SpatialDim must be 1, 2, or 3."); /*!< number of stress-strain terms */

    Omega_h::Matrix<mNumVoigtTerms,mNumVoigtTerms> mCellStiffnessCe;   /*!< cell stiffness matrix Ce, i.e. fourth-order material tensor for symmetric part of mesoscale strain */
    Omega_h::Matrix<mNumSkwTerms,mNumSkwTerms> mCellStiffnessCc;   /*!< cell stiffness matrix Cc, i.e. fourth-order material tensor for skew-symmetric part of mesoscale strain */
    Omega_h::Matrix<mNumVoigtTerms,mNumVoigtTerms> mCellStiffnessCm;   /*!< cell stiffness matrix Cm, i.e. fourth-order material tensor for symmetric part of micro-distortion */
    Omega_h::Vector<mNumVoigtTerms> mReferenceStrain;                /*!< reference strain tensor */

    Plato::Scalar mRayleighB; // stiffness coefficient

public:
    /******************************************************************************//**
     * \brief micromorphic Linear elastic material model constructor.
    **********************************************************************************/
    MicromorphicLinearElasticMaterial();

    /******************************************************************************//**
     * \brief micromorphic Linear elastic material model constructor.
     * \param [in] aParamList input parameter list
    **********************************************************************************/
    MicromorphicLinearElasticMaterial(const Teuchos::ParameterList& aParamList);

    /******************************************************************************//**
     * \brief Return cell stiffness matrix Ce.
     * \return cell stiffness matrix
    **********************************************************************************/
    decltype(mCellStiffnessCe)   getStiffnessMatrixCe() const {return mCellStiffnessCe;}

    /******************************************************************************//**
     * \brief Return cell stiffness matrix Cc.
     * \return cell stiffness matrix
    **********************************************************************************/
    decltype(mCellStiffnessCc)   getStiffnessMatrixCc() const {return mCellStiffnessCc;}

    /******************************************************************************//**
     * \brief Return cell stiffness matrix Cm.
     * \return cell stiffness matrix
    **********************************************************************************/
    decltype(mCellStiffnessCm)   getStiffnessMatrixCm() const {return mCellStiffnessCm;}

    /******************************************************************************//**
     * \brief Return reference strain tensor, i.e. homogenized strain tensor.
     * \return reference strain tensor
    **********************************************************************************/
    decltype(mReferenceStrain) getReferenceStrain() const {return mReferenceStrain;}

    decltype(mRayleighB)       getRayleighB()       const {return mRayleighB;}

private:
    /******************************************************************************//**
     * \brief Initialize member data to default values.
    **********************************************************************************/
    void initialize();

    /******************************************************************************//**
     * \brief Set reference strain tensor.
    **********************************************************************************/
    void setReferenceStrainTensor(const Teuchos::ParameterList& aParamList);
};
// class MicromorphicLinearElasticMaterial

}
// namespace Plato

