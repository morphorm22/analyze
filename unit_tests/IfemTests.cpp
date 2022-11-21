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

struct Range
{
private:
    std::string mPhysics = "undefined"; /*!< physics to be analyzed/simulated */
    std::unordered_map<std::string, Plato::ScalarMultiVector> mMV; /*!< map from data name to pod type */
    std::unordered_map<std::string, Plato::OrdinalType> mDataID2NumDofs; /*!< map from data name to number of degrees of freedom */
    std::unordered_map<std::string, std::vector<std::string>> mDataID2DofNames; /*!< map from data name to degrees of freedom names */

public:
    Range(){};
    Range(const std::string& aPhysics) : mPhysics(aPhysics)
    {}
    ~Range(){}

    std::vector<std::string> tags() const
    {
        std::vector<std::string> tTags;
        for(auto& tPair : mMV)
        {
            tTags.push_back(tPair.first);
        }
        return tTags;
    }
    Plato::ScalarMultiVector get(const std::string& aTag) const
    {
        auto tLowerTag = Plato::tolower(aTag);
        auto tItr = mMV.find(tLowerTag);
        if(tItr == mMV.end())
        {
            ANALYZE_THROWERR(std::string("Data with tag '") + aTag + "' is not defined in Range associative map")
        }
        return tItr->second;
    }
    void set(const std::string& aTag, const Plato::ScalarMultiVector& aData)
    {
        auto tLowerTag = Plato::tolower(aTag);
        mMV[tLowerTag] = aData;
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
    /***************************************************************************//**
     * \fn cycles
     * \brief Return number of solution cycles
     * \param number of solution cycles
     ******************************************************************************/
    Plato::OrdinalType cycles() const
    {
        if(this->empty())
        {
            ANALYZE_THROWERR("Range associative map is empty")
        }
        auto tTags = this->tags();
        const std::string tTag = tTags[0];
        auto tItr = mMV.find(tTag);
        auto tSolution = tItr->second;
        return tSolution.extent(0);
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
        if(mMV.empty())
        { return; }
        for(auto& tPair : mMV)
        { Plato::print_array_2D(tPair.second, tPair.first); }
    }
    bool empty() const
    {
        return mMV.empty();
    }
};
// struct Range

struct Domain
{
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
    virtual immersus::Range
    solution(const immersus::Domain & aDomain)=0;

    virtual Plato::Scalar
    criterionValue(const immersus::Domain& aDomain,const std::string& aName)=0;

    virtual Plato::ScalarVector
    criterionGradient(const immersus::Domain& aDomain,const std::string& aName)=0;
};
// class Abstract Problem

/******************************************************************************//**
 * \brief Manage scalar and vector function evaluations
**********************************************************************************/
template<typename PhysicsType>
class Problem: public immersus::AbstractProblem
{
public:
    virtual immersus::Range
    solution(const immersus::Domain & aDomain)
    { return immersus::Range(); }

    virtual Plato::Scalar
    criterionValue(const immersus::Domain& aDomain,const std::string& aName)
    { return 0; }

    virtual Plato::ScalarVector
    criterionGradient(const immersus::Domain & aDomain,const std::string& aName)
    {return Plato::ScalarVector("hello");}
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
