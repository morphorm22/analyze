/*
 * Solutions.hpp
 *
 *  Created on: Apr 5, 2021
 */

#pragma once

#include <vector>
#include <string>
#include <unordered_map>

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/***************************************************************************//**
 * \struct Solutions
 *  \brief Holds POD state solutions
 ******************************************************************************/
struct Solutions
{
private:
    std::string mPDE; /*!< partial differential equation constraint */
    std::string mPhysics; /*!< physics to be analyzed/simulated */
    std::unordered_map<std::string, Plato::ScalarMultiVector> mSolution; /*!< map from state solution name to 2D POD array */
    std::unordered_map<std::string, Plato::OrdinalType> mSolutionNameToNumDofsMap; /*!< map from state solution name to number of dofs */
    std::unordered_map<std::string, std::vector<std::string>> mSolutionNameToDofNamesMap; /*!< map from state solution name to dof names */

    std::unordered_map<std::string, Plato::ScalarArray3D> mSolutionArray3D; /*!< map from state solution name to 3D POD array */
    std::unordered_map<std::string, Plato::ScalarArray4D> mSolutionArray4D; /*!< map from state solution name to 4D POD array */

public:
    /***************************************************************************//**
     * \fn Solutions
     *
     * \brief Constructor.
     * \param [in] aPhysics physics to be analyzed/simulated
     * \param [in] aPDE     partial differential equation constraint type
     ******************************************************************************/
    explicit Solutions(std::string aPhysics = "undefined", std::string aPDE = "undefined");

    /***************************************************************************//**
     * \fn std::string pde
     *
     * \brief Return partial differential equation (pde) constraint type.
     * \return pde (string)
     ******************************************************************************/
    std::string pde() const;

    /***************************************************************************//**
     * \fn std::string physics
     *
     * \brief Return analyzed/simulated physics.
     * \return physics (string)
     ******************************************************************************/
    std::string physics() const;

    /***************************************************************************//**
     * \fn Plato::OrdinalType size
     *
     * \brief Return number of elements in solution map.
     * \return number of elements in solution map (integer)
     ******************************************************************************/
    Plato::OrdinalType size() const;

    /***************************************************************************//**
     * \fn std::vector<std::string> tags
     *
     * \brief Return list with state solution tags.
     * \return list with state solution tags
     ******************************************************************************/
    std::vector<std::string> tags() const;

    /***************************************************************************//**
     * \fn void set
     *
     * \brief Set value of an element in the solution map.
     * \param aTag  data tag
     * \param aData 2D POD array
     ******************************************************************************/
    void set(const std::string& aTag, const Plato::ScalarMultiVector& aData);

    /***************************************************************************//**
     * \fn void set
     *
     * \brief Set value of an element in the solution map.
     * \param aTag  data tag
     * \param aData 3D POD array
     ******************************************************************************/
    void set(const std::string& aTag, const Plato::ScalarArray3D& aData);

    /***************************************************************************//**
     * \fn void set
     *
     * \brief Set value of an element in the solution map.
     * \param aTag  data tag
     * \param aData 4D POD array
     ******************************************************************************/
    void set(const std::string& aTag, const Plato::ScalarArray4D& aData);

    /***************************************************************************//**
     * \fn void set
     *
     * \brief Set value of an element in the solution map.
     * \param aTag  data tag
     * \param aData 2D POD array
     * \param aDofNames list of dof names
     ******************************************************************************/
    void
    set( const std::string              & aTag, 
         const Plato::ScalarMultiVector & aData,
         const std::vector<std::string> & aDofNames );

    /***************************************************************************//**
     * \fn Plato::ScalarMultiVector get
     *
     * \brief Return 2D POD array.
     * \param aTag data tag
     ******************************************************************************/
    Plato::ScalarMultiVector get(const std::string& aTag) const;

    /***************************************************************************//**
     * \fn void get
     *
     * \brief Return 3D POD array.
     * \param aTag data tag
     * \param aData data
     ******************************************************************************/
    void get(const std::string& aTag, Plato::ScalarArray3D & aData) const;

    /***************************************************************************//**
     * \fn void get
     *
     * \brief Return 4D POD array.
     * \param aTag data tag
     * \param aData data
     ******************************************************************************/
    void get(const std::string& aTag, Plato::ScalarArray4D & aData) const;

    /***************************************************************************//**
     * \fn void set number of degrees of freedom (dofs) per node in map
     *
     * \brief Set value of an element in the solution-to-numdofs map.
     * \param aTag  data tag
     * \param aNumDofs number of dofs
     ******************************************************************************/
    void setNumDofs(const std::string& aTag, const Plato::OrdinalType& aNumDofs);

    /***************************************************************************//**
     * \fn void set names of degrees of freedom (dofs) for this entry in the map
     *
     * \brief Set value of an element in the solution-to-dofnames map.
     * \param aTag  data tag
     * \param aDofNames names of dofs
     ******************************************************************************/
    void setDofNames(const std::string& aTag, const std::vector<std::string>& aDofNames);

    /***************************************************************************//**
     * \fn Plato::OrdinalType get the number of degrees of freedom (dofs)
     *
     * \brief Return the number of dofs
     * \param aTag data tag
     ******************************************************************************/
    Plato::OrdinalType getNumDofs(const std::string& aTag) const;

    /***************************************************************************//**
     * \fn Plato::OrdinalType get the number of time steps
     *
     * \brief Return the number of time steps
     * \param aTag data tag
     ******************************************************************************/
    Plato::OrdinalType getNumTimeSteps() const;

    /***************************************************************************//**
     * \brief Return the names of the degrees of freedom
     * \param aTag data tag
     ******************************************************************************/
    std::vector<std::string> getDofNames(const std::string& aTag) const;

    /***************************************************************************//**
     * \fn void print
     *
     * \brief Print solutions metadata.
     ******************************************************************************/
    void print() const;

    /***************************************************************************//**
     * \fn bool defined
     *
     * \brief Check if solution with input tag is defined in the database
     * \param [in] aTag solution tag/identifier
     * \return boolean (true = is defined; false = not defined) 
     ******************************************************************************/
    bool defined(const std::string & aTag) const;

    /***************************************************************************//**
     * \fn bool empty
     *
     * \brief Check if the solution database is empty
     * \return boolean (true = is empty; false = is not empty) 
     ******************************************************************************/
    bool empty() const;
};
// struct Solutions

}
// namespace Plato
