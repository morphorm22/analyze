#ifndef ABSTRACT_VECTOR_FUNCTION_HYPERBOLIC_HPP
#define ABSTRACT_VECTOR_FUNCTION_HYPERBOLIC_HPP

#include "Solutions.hpp"

namespace Plato
{

namespace Hyperbolic
{

/******************************************************************************/
template<typename EvaluationType>
class AbstractVectorFunction
/******************************************************************************/
{
protected:
    const Plato::SpatialDomain & mSpatialDomain;

    Plato::DataMap& mDataMap;
    std::vector<std::string> mDofNames;
    std::vector<std::string> mDofDotNames;
    std::vector<std::string> mDofDotDotNames;

public:
    /******************************************************************************/
    explicit 
    AbstractVectorFunction(
        const Plato::SpatialDomain     & aSpatialDomain,
              Plato::DataMap           & aDataMap,
              std::vector<std::string>   aStateNames,
              std::vector<std::string>   aStateDotNames,
              std::vector<std::string>   aStateDotDotNames
    ) :
    /******************************************************************************/
        mSpatialDomain  (aSpatialDomain),
        mDataMap        (aDataMap),
        mDofNames       (aStateNames),
        mDofDotNames    (aStateDotNames),
        mDofDotDotNames (aStateDotDotNames)
    {
    }
    /******************************************************************************/
    virtual ~AbstractVectorFunction()
    /******************************************************************************/
    {
    }

    /****************************************************************************//**
    * \brief Return reference to mesh data base 
    ********************************************************************************/
    decltype(mSpatialDomain.Mesh) getMesh() const
    {
        return (mSpatialDomain.Mesh);
    }

    /****************************************************************************//**
    * \brief Return reference to state index map
    ********************************************************************************/
    const decltype(mDofNames)& getDofNames() const
    {
        return (mDofNames);
    }

    /****************************************************************************//**
    * \brief Return reference to state dot index map
    ********************************************************************************/
    const decltype(mDofDotNames)& getDofDotNames() const
    {
        return (mDofDotNames);
    }

    /****************************************************************************//**
    * \brief Return reference to state dot dot index map
    ********************************************************************************/
    const decltype(mDofDotDotNames)& getDofDotDotNames() const
    {
        return (mDofDotDotNames);
    }

    /**************************************************************************//**
    * \brief return the maximum eigenvalue of the gradient wrt state
    * \param [in] aConfig Current node locations
    * \return maximum eigenvalue
    ******************************************************************************/
    virtual Plato::Scalar getMaxEigenvalue(const Plato::ScalarArray3D & aConfig) const = 0;

    /**************************************************************************//**
    * \brief Call the output state function in the residual
    * \param [in] aSolutions State solutions database
    * \return output solutions database
    ******************************************************************************/
    virtual Plato::Solutions 
    getSolutionStateOutputData(const Plato::Solutions &aSolutions) const = 0;

    /******************************************************************************/
    virtual void
    evaluate(
        const Plato::ScalarMultiVectorT< typename EvaluationType::StateScalarType       > & aState,
        const Plato::ScalarMultiVectorT< typename EvaluationType::StateDotScalarType    > & aStateDot,
        const Plato::ScalarMultiVectorT< typename EvaluationType::StateDotDotScalarType > & aStateDotDot,
        const Plato::ScalarMultiVectorT< typename EvaluationType::ControlScalarType     > & aControl,
        const Plato::ScalarArray3DT    < typename EvaluationType::ConfigScalarType      > & aConfig,
              Plato::ScalarMultiVectorT< typename EvaluationType::ResultScalarType      > & aResult,
              Plato::Scalar aTimeStep = 0.0, 
              Plato::Scalar aCurrentTime = 0.0) const = 0;
    /******************************************************************************/

    /******************************************************************************/
    virtual void
    evaluate_boundary(
        const Plato::SpatialModel                                                         & aSpatialModel,
        const Plato::ScalarMultiVectorT< typename EvaluationType::StateScalarType       > & aState,
        const Plato::ScalarMultiVectorT< typename EvaluationType::StateDotScalarType    > & aStateDot,
        const Plato::ScalarMultiVectorT< typename EvaluationType::StateDotDotScalarType > & aStateDotDot,
        const Plato::ScalarMultiVectorT< typename EvaluationType::ControlScalarType     > & aControl,
        const Plato::ScalarArray3DT    < typename EvaluationType::ConfigScalarType      > & aConfig,
              Plato::ScalarMultiVectorT< typename EvaluationType::ResultScalarType      > & aResult,
              Plato::Scalar aTimeStep = 0.0, 
              Plato::Scalar aCurrentTime = 0.0) const = 0;
    /******************************************************************************/
};

} // namespace Hyperbolic

} // namespace Plato

#endif
