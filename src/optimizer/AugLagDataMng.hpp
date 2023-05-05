/*
 * AugLagDataMng.hpp
 *
 *  Created on: May 4, 2023
 */

#pragma once

#include <Teuchos_ParameterList.hpp>

#include "PlatoTypes.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Data manager for augmented Lagrangian data. This class is used in the 
 *   criteria using the augmented Lagrangian method. This class also holds the 
 *   tools responsible for updating the Lagrange multipliers and penalties. 
**********************************************************************************/
class AugLagDataMng
{
public:
    Plato::OrdinalType mNumCells = 0;               /*!< number of cells */
    Plato::OrdinalType mNumLocalConstraints = 0;    /*!< number of local constraints */
 
    Plato::Scalar mMaxPenalty = 10000.0;            /*!< maximum penalty value allowed for AL formulation */
    Plato::Scalar mInitiaPenalty = 1.0;             /*!< initial Lagrange multipliers */
    Plato::Scalar mPenaltyIncrement = 1.1;          /*!< increment multiplier for penalty values */
    Plato::Scalar mPenaltyUpdateParameter = 0.25;   /*!< previous constraint multiplier used for penaly update */
    Plato::Scalar mInitialLagrangeMultiplier = 0.0; /*!< initial Lagrange multipliers */

    Plato::ScalarVector mPenaltyValues;             /*!< penalty values for augmented Largangian formulation */
    Plato::ScalarVector mLocalMeasureLimits;        /*!< local constraint limit */
    Plato::ScalarVector mLagrangeMultipliers;       /*!< Lagrange multipliers for augmented Lagragian formulation */
    Plato::ScalarVector mCurrentConstraintValues;   /*!< current contraint values */
    Plato::ScalarVector mPreviousConstraintValues;  /*!< previous contraint values */

public:
    /******************************************************************************//**
     * \brief Constructor
    **********************************************************************************/
    AugLagDataMng(){}

    /******************************************************************************//**
     * \brief Destructor
    **********************************************************************************/
    ~AugLagDataMng(){}

    /******************************************************************************//**
     * \brief Initialize public member data
    **********************************************************************************/
    void initialize();

    /******************************************************************************//**
     * \brief Update Lagrange multipliers
    **********************************************************************************/
    void updateLagrangeMultipliers();

    /******************************************************************************//**
     * \brief Update penalty values 
    **********************************************************************************/
    void updatePenaltyValues();

    /******************************************************************************//**
     * \brief Parse numeric input parameters
     * \param [in] aParams teuchos parameter list
    **********************************************************************************/
    void parseNumerics(Teuchos::ParameterList &aParams);

    /******************************************************************************//**
     * \brief Parse limits on constraint criteria
     * \param [in] aParams teuchos parameter list
    **********************************************************************************/
    void parseLimits(Teuchos::ParameterList &aParams);

    /******************************************************************************//**
     * \brief Allocate memory for member containers
     * \param [in] aNumCells local number of cells
    **********************************************************************************/
    void allocateContainers(const Plato::OrdinalType &aNumCells);
};

}
// namespace Plato