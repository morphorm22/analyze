#ifndef ABSTRACT_VECTOR_FUNCTION_PARABOLIC_HPP
#define ABSTRACT_VECTOR_FUNCTION_PARABOLIC_HPP

#include "SpatialModel.hpp"
#include "PlatoStaticsTypes.hpp"
#include "Solutions.hpp"

namespace Plato
{

namespace Parabolic
{

/******************************************************************************/
template<typename EvaluationType>
class AbstractVectorFunction
/******************************************************************************/
{
    using StrVec = std::vector<std::string>;

protected:
    const Plato::SpatialDomain & mSpatialDomain;

    Plato::DataMap& mDataMap;
    StrVec mDofNames;
    StrVec mDofDotNames;

public:

    using AbstractType = Plato::Parabolic::AbstractVectorFunction<EvaluationType>;

    /******************************************************************************/
    explicit 
    AbstractVectorFunction(
        const Plato::SpatialDomain & aSpatialDomain,
              Plato::DataMap       & aDataMap
    ) :
    /******************************************************************************/
        mSpatialDomain (aSpatialDomain),
        mDataMap       (aDataMap)
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
    * \brief Return reference to state names
    ********************************************************************************/
    const decltype(mDofNames)& getDofNames() const
    {
        return (mDofNames);
    }

    /****************************************************************************//**
    * \brief Return reference to state dot names
    ********************************************************************************/
    const decltype(mDofDotNames)& getDofDotNames() const
    {
        return (mDofDotNames);
    }

    /****************************************************************************//**
    * \brief Pure virtual function to get output solution data
    ********************************************************************************/
    virtual Plato::Solutions getSolutionStateOutputData(const Plato::Solutions &aSolutions) const = 0;

    /******************************************************************************/
    virtual void
    evaluate(
        const Plato::ScalarMultiVectorT< typename EvaluationType::StateScalarType    > & aState,
        const Plato::ScalarMultiVectorT< typename EvaluationType::StateDotScalarType > & aStateDot,
        const Plato::ScalarMultiVectorT< typename EvaluationType::ControlScalarType  > & aControl,
        const Plato::ScalarArray3DT    < typename EvaluationType::ConfigScalarType   > & aConfig,
              Plato::ScalarMultiVectorT< typename EvaluationType::ResultScalarType   > & aResult,
              Plato::Scalar aTimeStep = 0.0) const = 0;
    /******************************************************************************/

    /******************************************************************************/
    virtual void
    evaluate_boundary(
        const Plato::SpatialModel                                                      & aModel,
        const Plato::ScalarMultiVectorT< typename EvaluationType::StateScalarType    > & aState,
        const Plato::ScalarMultiVectorT< typename EvaluationType::StateDotScalarType > & aStateDot,
        const Plato::ScalarMultiVectorT< typename EvaluationType::ControlScalarType  > & aControl,
        const Plato::ScalarArray3DT    < typename EvaluationType::ConfigScalarType   > & aConfig,
              Plato::ScalarMultiVectorT< typename EvaluationType::ResultScalarType   > & aResult,
              Plato::Scalar aTimeStep = 0.0) const = 0;
    /******************************************************************************/
};

} // namespace Parabolic

} // namespace Plato

#endif
