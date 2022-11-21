/*
 * IfemTests.cpp
 *
 *  Created on: Nov 21, 2022
 */

// c++ includes
#include <vector>
#include <unordered_map>

// trilinos includes
#include "Teuchos_UnitTestHarness.hpp"

// immersus includes
#include "MetaData.hpp"
#include "PlatoUtilities.hpp"
#include "PlatoStaticsTypes.hpp"

namespace immersus
{

template<typename Type>
struct Range
{
private:
    std::unordered_map<std::string, Type> mRange; /*!< map from data name to two-dimensional array of pod type */
    std::unordered_map<std::string, Plato::OrdinalType> mDataID2NumDofs; /*!< map from data name to number of degrees of freedom */
    std::unordered_map<std::string, std::vector<std::string>> mDataID2DofNames; /*!< map from data name to degrees of freedom names */

public:
    Range(){};
    ~Range(){}

    std::vector<std::string> tags() const
    {
        std::vector<std::string> tTags;
        for(auto& tPair : mRange)
        {
            tTags.push_back(tPair.first);
        }
        return tTags;
    }
    Type get(const std::string& aTag) const
    {
        auto tLowerTag = Plato::tolower(aTag);
        auto tItr = mRange.find(tLowerTag);
        if(tItr == mRange.end())
        {
            ANALYZE_THROWERR(std::string("Data with tag '") + aTag + "' is not defined in Range associative map")
        }
        return tItr->second;
    }
    void set(const std::string& aTag, const Type& aData)
    {
        auto tLowerTag = Plato::tolower(aTag);
        mRange[tLowerTag] = aData;
    }
    Plato::OrdinalType dofs(const std::string& aTag) const
    {
        auto tLowerTag = Plato::tolower(aTag);
        auto tItr = mDataID2NumDofs.find(tLowerTag);
        if(tItr == mDataID2NumDofs.end())
        {
            ANALYZE_THROWERR(std::string("Data with tag '") + aTag + "' is not defined in Range associative map")
        }
        return tItr->second;
    }
    std::vector<std::string> dof_names(const std::string& aTag) const
    {
        auto tLowerTag = Plato::tolower(aTag);
        auto tItr = mDataID2DofNames.find(tLowerTag);
        if(tItr == mDataID2DofNames.end())
        {
            return std::vector<std::string>(0);
        }
        return tItr->second;
    }
    void print() const
    {
        if(mRange.empty())
        { return; }
        for(auto& tPair : mRange)
        { Plato::print_array_2D(tPair.second, tPair.first); }
    }
    bool empty() const
    {
        return mRange.empty();
    }
};
// struct Range

struct Domain
{
    std::unordered_map<std::string,std::string> arguments; /*!< map to function-related arguments, e.g., derivative type */
    std::unordered_map<std::string,Plato::Scalar> scalars; /*!< map to scalar quantities of interest */
    std::unordered_map<std::string,Plato::ScalarVector> vectors; /*!< map to scalar quantities of interest */
};

class AbstractProblem
{
public:
    virtual ~AbstractProblem(){}

    /******************************************************************************//**
     * \brief Write results to output database.
     * \param [in] aFilename name of output database file
    **********************************************************************************/
    virtual void output(const std::string& aFilename) = 0;

    /******************************************************************************//**
     * \brief Solve numerical simulation
     * \param [in] aDomain independent variables
     * \return dependent variables
    **********************************************************************************/
    virtual immersus::Range<Plato::ScalarMultiVector>
    solution(const immersus::Domain & aDomain)=0;
};
// class Abstract Problem

/******************************************************************************//**
 * \brief Manage scalar and vector function evaluations
**********************************************************************************/
template<typename PhysicsType>
class Problem: public immersus::AbstractProblem
{
public:
    void output(const std::string& aFilename){}

    immersus::Range<Plato::ScalarMultiVector>
    solution(const immersus::Domain & aDomain){ return Range<Plato::ScalarMultiVector>(); }
};
// class Problem

}
// namespace immersus

namespace IfemTests
{

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_NewtonRaphsonStoppingCriterion)
{
    std::vector<double> tVector;
}

}
// namespace IfemTests
