#pragma once

#include "Solutions.hpp"

namespace Plato
{

namespace Hyperbolic
{

template<typename EvaluationType>
class AbstractVectorFunction
{
protected:
    const Plato::SpatialDomain & mSpatialDomain;

    Plato::DataMap& mDataMap;
    std::vector<std::string> mDofNames;
    std::vector<std::string> mDofDotNames;
    std::vector<std::string> mDofDotDotNames;

public:

    using AbstractType = typename Plato::Hyperbolic::AbstractVectorFunction<EvaluationType>;

    explicit 
    AbstractVectorFunction(
        const Plato::SpatialDomain     & aSpatialDomain,
              Plato::DataMap           & aDataMap,
              std::vector<std::string>   aStateNames,
              std::vector<std::string>   aStateDotNames,
              std::vector<std::string>   aStateDotDotNames
    ) :
        mSpatialDomain  (aSpatialDomain),
        mDataMap        (aDataMap),
        mDofNames       (aStateNames),
        mDofDotNames    (aStateDotNames),
        mDofDotDotNames (aStateDotDotNames)
    {
    }
    virtual ~AbstractVectorFunction()
    {
    }

    decltype(mSpatialDomain.Mesh) getMesh() const
    {
        return (mSpatialDomain.Mesh);
    }

    const decltype(mDofNames)& getDofNames() const
    {
        return (mDofNames);
    }

    const decltype(mDofDotNames)& getDofDotNames() const
    {
        return (mDofDotNames);
    }

    const decltype(mDofDotDotNames)& getDofDotDotNames() const
    {
        return (mDofDotDotNames);
    }

    virtual Plato::Scalar getMaxEigenvalue(const Plato::ScalarArray3D & aConfig) const = 0;

    virtual Plato::Solutions 
    getSolutionStateOutputData(const Plato::Solutions &aSolutions) const = 0;

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
};

} // namespace Hyperbolic

} // namespace Plato
