/*
 * NeumannBC.hpp
 *
 *  Created on: Mar 15, 2020
 */

#pragma once

#include <sstream>

#include <Teuchos_ParameterList.hpp>

#include "AnalyzeMacros.hpp"
#include "PlatoMathExpr.hpp"
#include "PlatoUtilities.hpp"
#include "bcs/neumann/NeumannForce.hpp"
#include "bcs/neumann/NeumannPressure.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Natural boundary condition type ENUM
*******************************************************************************/
struct Neumann
{
    enum bc_t
    {
        UNDEFINED = 0,
        UNIFORM = 1,
        UNIFORM_PRESSURE = 2,
        UNIFORM_COMPONENT = 3,
    };
};
// struct Neumann

/***************************************************************************//**
 * \brief Return natural boundary condition type
 * \param [in] aType natural boundary condition type string
 * \return natural boundary condition type enum
*******************************************************************************/
inline Plato::Neumann::bc_t natural_boundary_condition_type(const std::string& aType)
{
    auto tLowerTag = Plato::tolower(aType);
    if(tLowerTag == "uniform")
    {
        return Plato::Neumann::UNIFORM;
    }
    else if(tLowerTag == "uniform pressure")
    {
        return Plato::Neumann::UNIFORM_PRESSURE;
    }
    else if(tLowerTag == "uniform component")
    {
        return Plato::Neumann::UNIFORM_COMPONENT;
    }
    else
    {
        ANALYZE_THROWERR(std::string("Natural Boundary Condition: 'Type' Parameter Keyword: '") + tLowerTag + "' is not supported.")
    }
}
// function natural_boundary_condition_type

/***************************************************************************//**
 * \brief Class for natural boundary conditions.
 *
 * \tparam ElementType  Element type
 * \tparam DofsPerNode  number degrees of freedom per node
 * \tparam DofOffset    degrees of freedom offset
 *
*******************************************************************************/
template<
  typename ElementType,
  Plato::OrdinalType NumDofs=ElementType::mNumSpatialDims,
  Plato::OrdinalType DofsPerNode=NumDofs,
  Plato::OrdinalType DofOffset=0>
class NeumannBC
{
    const std::string mName;         /*!< user-defined load sublist name */
    const std::string mType;         /*!< natural boundary condition type */
    const std::string mSideSetName;  /*!< side set name */
    Plato::Array<NumDofs> mFlux;  /*!< force vector values */
    std::shared_ptr<Plato::MathExpr> mFluxExpr[NumDofs];

public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aLoadName user-defined name for natural boundary condition sublist
     * \param [in] aSubList  natural boundary condition input parameter sublist
    *******************************************************************************/
    NeumannBC<ElementType, NumDofs, DofsPerNode, DofOffset>(const std::string & aLoadName, Teuchos::ParameterList &aSubList) :
        mName(aLoadName),
        mType(aSubList.get<std::string>("Type")),
        mSideSetName(aSubList.get<std::string>("Sides")),
        mFluxExpr{nullptr}
    {
        auto tIsValue = aSubList.isType<Teuchos::Array<Plato::Scalar>>("Vector");
        auto tIsExpr  = aSubList.isType<Teuchos::Array<std::string>>("Vector");

        if (tIsValue)
        {
            auto tFlux = aSubList.get<Teuchos::Array<Plato::Scalar>>("Vector");
            for(Plato::OrdinalType tDof=0; tDof<NumDofs; tDof++)
            {
                mFlux(tDof) = tFlux[tDof];
            }
        }
        else
        if (tIsExpr)
        {
            auto tExpr = aSubList.get<Teuchos::Array<std::string>>("Vector");
            for(Plato::OrdinalType tDof=0; tDof<NumDofs; tDof++)
            {
                mFluxExpr[tDof] = std::make_shared<Plato::MathExpr>(tExpr[tDof]);
                mFlux(tDof) = mFluxExpr[tDof]->value(0.0);
            }
        }
    }

    /***************************************************************************//**
     * \brief Destructor
    *******************************************************************************/
    ~NeumannBC(){}

    /***************************************************************************//**
     * \brief Get the contribution to the assembled forcing vector.
     *
     * \tparam StateScalarType   state forward automatically differentiated (FAD) type
     * \tparam ControlScalarType control FAD type
     * \tparam ConfigScalarType  configuration FAD type
     * \tparam ResultScalarType  result FAD type
     *
     * \param [in]  aState        2-D view of state variables.
     * \param [in]  aControl      2-D view of control variables.
     * \param [in]  aConfig       3-D view of configuration variables.
     * \param [out] aResult       Assembled vector to which the boundary terms will be added
     * \param [in]  aScale        scalar multiplier
     *
     * The boundary terms are integrated on the parameterized surface, \f$\phi(\xi,\psi)\f$, according to:
     *  \f{eqnarray*}{
     *    \phi(\xi,\psi)=
     *       \left\{
     *        \begin{array}{ccc}
     *          N_I\left(\xi,\psi\right) x_I &
     *          N_I\left(\xi,\psi\right) y_I &
     *          N_I\left(\xi,\psi\right) z_I
     *        \end{array}
     *       \right\} \\
     *     f^{el}_{Ii} = \int_{\partial\Omega_{\xi}} N_I\left(\xi,\psi\right) t_i
     *          \left|\left|
     *            \frac{\partial\phi}{\partial\xi} \times \frac{\partial\phi}{\partial\psi}
     *          \right|\right| d\xi d\psi
     * \f}
    *******************************************************************************/
    template<typename StateScalarType,
             typename ControlScalarType,
             typename ConfigScalarType,
             typename ResultScalarType>
    void get(const Plato::SpatialModel&,
             const Plato::ScalarMultiVectorT<  StateScalarType>&,
             const Plato::ScalarMultiVectorT<ControlScalarType>&,
             const Plato::ScalarArray3DT    < ConfigScalarType>&,
             const Plato::ScalarMultiVectorT< ResultScalarType>&,
             Plato::Scalar aScale,
             Plato::Scalar aCurrentTime);

    /***************************************************************************//**
     * \brief Return natural boundary condition sublist name
     * \return sublist name
    *******************************************************************************/
    decltype(mName) const& getSubListName() const { return mName; }

    /***************************************************************************//**
     * \brief Return side set name for this natural boundary condition
     * \return side set name
    *******************************************************************************/
    decltype(mSideSetName) const& getSideSetName() const { return mSideSetName; }

}; // class NeumannBC

/***************************************************************************//**
 * \brief NeumannBC::get function definition
*******************************************************************************/
template<typename ElementType, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
template<typename StateScalarType,
         typename ControlScalarType,
         typename ConfigScalarType,
         typename ResultScalarType>
void NeumannBC<ElementType, NumDofs, DofsPerNode, DofOffset>::get(
    const Plato::SpatialModel                          & aSpatialModel,
    const Plato::ScalarMultiVectorT<  StateScalarType> & aState,
    const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
    const Plato::ScalarArray3DT    < ConfigScalarType> & aConfig,
    const Plato::ScalarMultiVectorT< ResultScalarType> & aResult,
          Plato::Scalar aScale,
          Plato::Scalar aCurrentTime
)
{

    for(int iDim=0; iDim<NumDofs; iDim++)
    {
        if(mFluxExpr[iDim])
        {
            mFlux(iDim) = mFluxExpr[iDim]->value(aCurrentTime);
        }
    }

    auto tType = Plato::natural_boundary_condition_type(mType);
    switch(tType)
    {
        case Plato::Neumann::UNIFORM:
        case Plato::Neumann::UNIFORM_COMPONENT:
        {
            Plato::NeumannForce<ElementType, NumDofs, DofsPerNode, DofOffset> tSurfaceLoad(mSideSetName, mFlux);
            tSurfaceLoad(aSpatialModel, aState, aControl, aConfig, aResult, aScale);
            break;
        }
        case Plato::Neumann::UNIFORM_PRESSURE:
        {
             Plato::NeumannPressure<ElementType, NumDofs, DofsPerNode, DofOffset> tSurfacePress(mSideSetName, mFlux);
             tSurfacePress(aSpatialModel, aState, aControl, aConfig, aResult, aScale);
            break;
        }
        default:
        {
            std::stringstream tMsg;
            tMsg << "Natural Boundary Condition: Natural Boundary Condition Type '" << mType.c_str() << "' is NOT supported.";
            ANALYZE_THROWERR(tMsg.str().c_str())
        }
    }
}

}
// namespace Plato
