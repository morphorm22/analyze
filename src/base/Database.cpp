/*
 * Database.cpp
 *
 *  Created on: Apr 6, 2021
 */

#include "BLAS1.hpp"
#include "PlatoUtilities.hpp"
#include "base/Database.hpp"

namespace Plato
{

Plato::Scalar 
Database::
scalar(
  const std::string& aTag
) const
{
    auto tLowerTag = Plato::tolower(aTag);
    auto tItr = mScalars.find(tLowerTag);
    if(tItr == mScalars.end())
    {
        ANALYZE_THROWERR(std::string("Scalar with tag '") + aTag + "' is not defined in the variables map.")
    }
    return tItr->second;
}

void 
Database::
scalar(
  const std::string& aTag, 
  const Plato::Scalar& aInput
)
{
    auto tLowerTag = Plato::tolower(aTag);
    mScalars[tLowerTag] = aInput;
}

Plato::ScalarVector 
Database::
vector(
  const std::string& aTag
) const
{
    auto tLowerTag = Plato::tolower(aTag);
    auto tItr = mVectors.find(tLowerTag);
    if(tItr == mVectors.end())
    {
        ANALYZE_THROWERR(std::string("Vector with tag '") + aTag + "' is not defined in the variables map.")
    }
    return tItr->second;
}

void 
Database::
vector(
  const std::string& aTag, 
  const Plato::ScalarVector& aInput
)
{
    auto tLowerTag = Plato::tolower(aTag);
    mVectors[tLowerTag] = aInput;
}

bool 
Database::
isVectorMapEmpty() const
{
    return mVectors.empty();
}

bool 
Database::
isScalarMapEmpty() const
{
    return mScalars.empty();
}

bool 
Database::
isScalarVectorDefined(
  const std::string & aTag
) const
{
    auto tLowerTag = Plato::tolower(aTag);
    auto tVectorMapItr = mVectors.find(tLowerTag);
    auto tFoundTag = tVectorMapItr != mVectors.end();

    if(tFoundTag)
    { return true; }
    else
    { return false; }
}

bool 
Database::
isScalarDefined(
  const std::string & aTag
) const
{
    auto tLowerTag = Plato::tolower(aTag);
    auto tScalarMapItr = mScalars.find(tLowerTag);
    auto tFoundTag = tScalarMapItr != mScalars.end();

    if(tFoundTag)
    { return true; }
    else
    { return false; }
}

bool 
Database::
defined(
  const std::string & aTag
) const
{
    auto tLowerTag = Plato::tolower(aTag);
    auto tScalarMapItr = mScalars.find(tLowerTag);
    auto tFoundScalarTag = tScalarMapItr != mScalars.end();
    auto tVectorMapItr = mVectors.find(tLowerTag);
    auto tFoundVectorTag = tVectorMapItr != mVectors.end();

    if(tFoundScalarTag || tFoundVectorTag)
    { return true; }
    else
    { return false; }
}

void 
Database::
print() const
{
    this->printScalarMap();
    this->printVectorMap();
}

void 
Database::
printVectorMap() const
{
    if(mVectors.empty())
    {
        return;
    }

    std::cout << "Print Vector Map\n";
    for(auto& tPair : mVectors)
    {
        std::cout << "name = " << tPair.first << ", norm = " << Plato::blas1::norm(tPair.second) << "\n" << std::flush;
    }
}

void 
Database::
printScalarMap() const
{
    if(mScalars.empty())
    {
        return;
    }

    std::cout << "Print Scalar Map\n";
    for(auto& tPair : mScalars)
    {
        std::cout << "name = " << tPair.first << ", value = " << tPair.second << "\n" << std::flush;
    }
}

}
// namespace Plato

namespace Plato
{

void FieldTags::set(const std::string& aTag, const std::string& aID)
{
    mFields[aTag] = aID;
}

std::vector<std::string> FieldTags::tags() const
{
    std::vector<std::string> tTags;
    for(auto& tPair : mFields)
    {
        tTags.push_back(tPair.first);
    }
    return tTags;
}

std::string FieldTags::id(const std::string& aTag) const
{
    auto tItr = mFields.find(aTag);
    if(tItr == mFields.end())
    {
        ANALYZE_THROWERR(std::string("Field with tag '") + aTag + "' is not defined.")
    }
    return tItr->second;
}

}
// namespace Plato
