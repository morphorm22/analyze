#pragma once

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

namespace Elliptic
{

template<typename PhysicsType>
void
CriterionEvaluatorSolutionFunction<PhysicsType>::
initialize (
    Teuchos::ParameterList & aProblemParams
)
{
  auto tFunctionParams = aProblemParams.sublist("Criteria").sublist(mFunctionName);
  mDomainName = tFunctionParams.get<std::string>("Domain");
  mMagnitudeSpecified = tFunctionParams.get<bool>("Magnitude", false);
  mNormalSpecified = tFunctionParams.isType<Teuchos::Array<Plato::Scalar>>("Normal");
  mTargetSolutionVectorSpecified = tFunctionParams.isType<Teuchos::Array<Plato::Scalar>>("TargetSolutionVector");
  mTargetSolutionSpecified = tFunctionParams.isType<Plato::Scalar>("TargetSolution");
  mTargetMagnitudeSpecified = tFunctionParams.isType<Plato::Scalar>("TargetMagnitude");
  mSolutionType = solution_type_t::UNKNOWN_TYPE;
  
  auto tNumDofsPerNode = mNumDofsPerNode;
  if(mTargetSolutionVectorSpecified)
  {
    // user only needs to specify the target solution vector
    mSolutionType = solution_type_t::DIFF_BETWEEN_SOLUTION_VECTOR_AND_TARGET_VECTOR;
  }
  else if(mTargetMagnitudeSpecified)
  {
    // user must specify target magnitude and normal vector
    if(!mNormalSpecified)
    {
      ANALYZE_THROWERR("Parsing 'Solution' criterion:  'Normal' must be specified in addition to 'TargetMagnitude' in order to know what direction to consider.");
    }
    mSolutionType = solution_type_t::DIFF_BETWEEN_SOLUTION_MAG_IN_DIRECTION_AND_TARGET;
  }
  else if(mTargetSolutionSpecified)
  {
    // user must specify target solution value and normal vector
    if(!mNormalSpecified)
    {
      ANALYZE_THROWERR("Parsing 'Solution' criterion:  'Normal' must be specified in addition to 'TargetSolution' in order to know what direction to consider.");
    }
    mSolutionType = solution_type_t::DIFF_BETWEEN_SOLUTION_IN_DIRECTION_AND_TARGET_SOLUTION_IN_DIRECTION;
  }
  else if(mMagnitudeSpecified)
  {
    // user must specify a direction that the magnitude will be measured in
    if(!mNormalSpecified)
    {
      ANALYZE_THROWERR("Parsing 'Solution' criterion:  'Normal' must be specified in addition to 'Magnitude' in order to know what direction to consider.");
    }
    mSolutionType = solution_type_t::SOLUTION_MAG_IN_DIRECTION;
  }
  else if(mNormalSpecified)
  {
    mSolutionType = solution_type_t::SOLUTION_IN_DIRECTION;
  }
  if(mSolutionType == solution_type_t::UNKNOWN_TYPE)
  {
    ANALYZE_THROWERR("Parsing 'Solution' criterion:  Could not determine solution type based on user-provided input.");
  }

  switch (mSolutionType)
  {
    case solution_type_t::DIFF_BETWEEN_SOLUTION_VECTOR_AND_TARGET_VECTOR:
      {
        initialize_target_vector(tFunctionParams);
        auto tTargetSolutionVector = mTargetSolutionVector;
        std::stringstream ss;
        ss << "Solution Type: Measure the magnitude of the difference between the solution vector and the user-specified target vector." << std::endl;
        ss << "Target Vector:";
        for(int h=0; h<tNumDofsPerNode; ++h)
        {
          ss << " " << tTargetSolutionVector[h];
        }
        ss << std::endl;
        REPORT(ss.str());
      }
      break; 
    case solution_type_t::DIFF_BETWEEN_SOLUTION_MAG_IN_DIRECTION_AND_TARGET:
      {
        mTargetMagnitude = tFunctionParams.get<Plato::Scalar>("TargetMagnitude");
        initialize_normal_vector(tFunctionParams);
        auto tTargetMagnitude = mTargetMagnitude;
        auto tNormal = mNormal;
        std::stringstream ss;
        ss << "Solution Type: Measure the difference between the solution magnitude in the specified direction and the target magnitude in that direction." << std::endl;
        ss << "Normal Vector:";         
        for(int h=0; h<tNumDofsPerNode; ++h){
          ss << " " << tNormal[h];
        }
        ss << std::endl;
        ss << "Target Magnitude: " << tTargetMagnitude << std::endl;
        REPORT(ss.str());
      }
      break;
    case solution_type_t::SOLUTION_MAG_IN_DIRECTION:
      {
        initialize_normal_vector(tFunctionParams);
        auto tNormal = mNormal;
        std::stringstream ss;
        ss << "Solution Type: Measure the magnitude of the solution in the specified direction." << std::endl;
        ss << "Normal Vector:";
        for(int h=0; h<tNumDofsPerNode; ++h)
        {
          ss << " " << tNormal[h];
        }
        ss << std::endl;
        REPORT(ss.str());
      }
      break;
    case solution_type_t::SOLUTION_IN_DIRECTION:
      {
        initialize_normal_vector(tFunctionParams);
        auto tNormal = mNormal;
        std::stringstream ss;
        ss << "Solution Type: Measure the solution in the specified direction." << std::endl;
        ss << "Normal Vector:";
        for(int h=0; h<tNumDofsPerNode; ++h)
        {
          ss << " " << tNormal[h];
        }
        ss << std::endl;
        REPORT(ss.str());
      }
      break; 
    case solution_type_t::DIFF_BETWEEN_SOLUTION_IN_DIRECTION_AND_TARGET_SOLUTION_IN_DIRECTION:
      {
        mTargetSolution = tFunctionParams.get<Plato::Scalar>("TargetSolution");
        initialize_normal_vector(tFunctionParams);
        auto tNormal = mNormal;
        auto tTargetSolution = mTargetSolution;
        std::stringstream ss;
        ss << "Solution Type: Measure the solution in the specified direction." << std::endl;
        ss << "Normal Vector:";
        for(int h=0; h<tNumDofsPerNode; ++h)
        {
          ss << " " << tNormal[h];
        }
        ss << std::endl;
        ss << "Target Solution: " << tTargetSolution << std::endl;
        REPORT(ss.str());
      }
      break; 
  }
}
  
template<typename PhysicsType>
void
CriterionEvaluatorSolutionFunction<PhysicsType>::
initialize_target_vector(
  Teuchos::ParameterList &aFunctionParams
)
{
  if (aFunctionParams.isType<Teuchos::Array<Plato::Scalar>>("TargetSolutionVector") == false)
  {
    if (mNumDofsPerNode != 1)
    {
        ANALYZE_THROWERR("Parsing 'Solution' criterion:  'TargetSolutionVector' parameter missing.");
    }
    else
    {
        mTargetSolutionVector[0] = 1.0;
    }
  }
  else
  {
    auto tTargetArray = aFunctionParams.get<Teuchos::Array<Plato::Scalar>>("TargetSolutionVector");
    if(tTargetArray.size() > mNumDofsPerNode)
    {
      std::stringstream ss;
      ss << "Extra terms in 'TargetSolutionVector' array." << std::endl;
      ss << "Number of terms provided: " << tTargetArray.size() << std::endl;
      ss << "Expected number of dofs per node (" << mNumDofsPerNode << ")." << std::endl;
      ss << "Ignoring extra terms." << std::endl;
      REPORT(ss.str());
      for(int i=0; i<mNumDofsPerNode; i++)
      {
        mTargetSolutionVector[i] = tTargetArray[i];
      }
    }
    else
    if(tTargetArray.size() < mNumDofsPerNode)
    {
      std::stringstream ss;
      ss << "'TargetSolutionVector' array is missing terms." << std::endl;
      ss << "Number of terms provided: " << tTargetArray.size() << std::endl;
      ss << "Expected number of dofs per node (" << mNumDofsPerNode << ")." << std::endl;
      ss << "Missing terms will be set to zero." << std::endl;
      REPORT(ss.str());
      for(int i=0; i<tTargetArray.size(); i++)
      {
        mTargetSolutionVector[i] = tTargetArray[i];
      }
    }
    else
    {
      for(int i=0; i<tTargetArray.size(); i++)
      {
        mTargetSolutionVector[i] = tTargetArray[i];
      }
    }
  }
}

template<typename PhysicsType>
void
CriterionEvaluatorSolutionFunction<PhysicsType>::
initialize_normal_vector(
  Teuchos::ParameterList &aFunctionParams
)
{
  if (aFunctionParams.isType<Teuchos::Array<Plato::Scalar>>("Normal") == false)
  {
    if (mNumDofsPerNode != 1)
    {
      ANALYZE_THROWERR("Parsing 'Solution' criterion:  'Normal' parameter missing.");
    }
    else
    {
      mNormal[0] = 1.0;
    }
  }
  else
  {
    auto tNormalArray = aFunctionParams.get<Teuchos::Array<Plato::Scalar>>("Normal");
    if(tNormalArray.size() > mNumDofsPerNode)
    {
      std::stringstream ss;
      ss << "Extra terms in 'Normal' array." << std::endl;
      ss << "Number of terms provided: " << tNormalArray.size() << std::endl;
      ss << "Expected number of dofs per node (" << mNumDofsPerNode << ")." << std::endl;
      ss << "Ignoring extra terms." << std::endl;
      REPORT(ss.str());
      for(int i=0; i<mNumDofsPerNode; i++)
      {
        mNormal[i] = tNormalArray[i];
      }
    }
    else
    if(tNormalArray.size() < mNumDofsPerNode)
    {
      std::stringstream ss;
      ss << "'Normal' array is missing terms." << std::endl;
      ss << "Number of terms provided: " << tNormalArray.size() << std::endl;
      ss << "Expected number of dofs per node (" << mNumDofsPerNode << ")." << std::endl;
      ss << "Missing terms will be set to zero." << std::endl;
      REPORT(ss.str());
      for(int i=0; i<tNormalArray.size(); i++)
      {
        mNormal[i] = tNormalArray[i];
      }
    }
    else
    {
      for(int i=0; i<tNormalArray.size(); i++)
      {
        mNormal[i] = tNormalArray[i];
      }
    }
  }
}

template<typename PhysicsType>
CriterionEvaluatorSolutionFunction<PhysicsType>::
CriterionEvaluatorSolutionFunction(
  const Plato::SpatialModel    & aSpatialModel,
        Plato::DataMap         & aDataMap,
        Teuchos::ParameterList & aProblemParams,
        std::string            & aName
) :
  mSpatialModel (aSpatialModel),
  mFunctionName (aName),
  mNormal{0.0}
{
  initialize(aProblemParams);
}

template<typename PhysicsType>
bool 
CriterionEvaluatorSolutionFunction<PhysicsType>::
isLinear() 
const
{
  return false;
}

template<typename PhysicsType>
Plato::Scalar
CriterionEvaluatorSolutionFunction<PhysicsType>::
  value(const Plato::Database & aDatabase,
        const Plato::Scalar   & aCycle
) const
{
  auto tStates = aDatabase.vector("states");

  auto tNodeIds = mSpatialModel.Mesh->GetNodeSetNodes(mDomainName);
  auto tNumNodes = tNodeIds.size();

  auto tNormal = mNormal;
  auto tTargetSolutionVector = mTargetSolutionVector;
  auto tTargetSolution = mTargetSolution;
  auto tTargetMagnitude = mTargetMagnitude;
  auto tNumDofsPerNode = mNumDofsPerNode;

  Scalar tReturnValue(0.0);
  switch (mSolutionType)
  {
    case solution_type_t::DIFF_BETWEEN_SOLUTION_VECTOR_AND_TARGET_VECTOR:
      {
        Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0, tNumNodes), 
        KOKKOS_LAMBDA(const Plato::OrdinalType& aNodeOrdinal, Scalar & aLocalValue)
        {
          auto tIndex = tNodeIds[aNodeOrdinal];
          Plato::Scalar ds(0.0);
          for(int iDof=0; iDof<tNumDofsPerNode; iDof++)
          {
            auto dv = tStates(tNumDofsPerNode*tIndex+iDof);
            ds += (dv-tTargetSolutionVector[iDof])*(dv-tTargetSolutionVector[iDof]);
          }
          ds = (ds > 0.0) ? sqrt(ds) : ds;
          aLocalValue += ds;
        }, tReturnValue);
        tReturnValue /= tNumNodes;
        std::stringstream ss;
        ss << "Magnitude of the difference vector between actual solution and target solution vectors: " 
           << tReturnValue << std::endl;
        REPORT(ss.str());
      }
      break; 
    case solution_type_t::DIFF_BETWEEN_SOLUTION_MAG_IN_DIRECTION_AND_TARGET:
      {
        Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0, tNumNodes), 
        KOKKOS_LAMBDA(const Plato::OrdinalType& aNodeOrdinal, Scalar & aLocalValue)
        {
          auto tIndex = tNodeIds[aNodeOrdinal];
          Plato::Scalar ds(0.0);
          for(int iDof=0; iDof<tNumDofsPerNode; iDof++)
          {
            auto dv = tStates(tNumDofsPerNode*tIndex+iDof);
            ds += (tNormal[iDof]*dv) * (tNormal[iDof]*dv);
          }
          ds = (ds > 0.0) ? sqrt(ds) : ds;  
          aLocalValue += ds;
        }, tReturnValue);
        tReturnValue /= tNumNodes;
        tReturnValue = fabs(tReturnValue - tTargetMagnitude); 
        std::stringstream ss;
        ss << "Absolute value of difference between actual magnitude and target magnitude: " << tReturnValue << std::endl;
        REPORT(ss.str());
      }
      break;
    case solution_type_t::SOLUTION_MAG_IN_DIRECTION:
      {
        Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0, tNumNodes), 
        KOKKOS_LAMBDA(const Plato::OrdinalType& aNodeOrdinal, Scalar & aLocalValue)
        {
          auto tIndex = tNodeIds[aNodeOrdinal];
          Plato::Scalar ds(0.0);
          for(int iDof=0; iDof<tNumDofsPerNode; iDof++)
          {
            auto dv = tStates(tNumDofsPerNode*tIndex+iDof);
            ds += (tNormal[iDof]*dv) * (tNormal[iDof]*dv);
          }
          ds = (ds > 0.0) ? sqrt(ds) : ds;  
          aLocalValue += ds;
        }, tReturnValue);
        tReturnValue /= tNumNodes;
        std::stringstream ss;
        ss << "Magnitude of solution in given direction: " << tReturnValue << std::endl;
        REPORT(ss.str());
      }
      break;
    case solution_type_t::SOLUTION_IN_DIRECTION:
      {
        Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0, tNumNodes), 
        KOKKOS_LAMBDA(const Plato::OrdinalType& aNodeOrdinal, Scalar & aLocalValue)
        {
          auto tIndex = tNodeIds[aNodeOrdinal];
          for(int iDof=0; iDof<tNumDofsPerNode; iDof++)
          {
            auto dv = tStates(tNumDofsPerNode*tIndex+iDof);
            aLocalValue += (tNormal[iDof]*dv);
          }
        }, tReturnValue);
        tReturnValue /= tNumNodes;
        std::stringstream ss;
        ss << "Solution in given direction: " << tReturnValue << std::endl;
        REPORT(ss.str());
      }
      break; 
    case solution_type_t::DIFF_BETWEEN_SOLUTION_IN_DIRECTION_AND_TARGET_SOLUTION_IN_DIRECTION:
      {
        Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0, tNumNodes), 
        KOKKOS_LAMBDA(const Plato::OrdinalType& aNodeOrdinal, Scalar & aLocalValue)
        {
          auto tIndex = tNodeIds[aNodeOrdinal];
          for(int iDof=0; iDof<tNumDofsPerNode; iDof++)
          {
            auto dv = tStates(tNumDofsPerNode*tIndex+iDof);
            aLocalValue += (tNormal[iDof]*dv);
          }
        }, tReturnValue);
        tReturnValue /= tNumNodes;
        tReturnValue = fabs(tReturnValue - tTargetSolution); 
        std::stringstream ss;
        ss << "Absolute value of difference between current solution in given direction and target solution in given direction: " << tReturnValue << std::endl;
        REPORT(ss.str());
      }
      break; 
  }
  return tReturnValue;
}

template<typename PhysicsType>
Plato::ScalarVector
CriterionEvaluatorSolutionFunction<PhysicsType>::
gradientConfig(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle
) const
{
  const Plato::OrdinalType tNumNodes = mSpatialModel.Mesh->NumNodes();
  const Plato::OrdinalType tNumDofs = mNumSpatialDims * tNumNodes;
  Plato::ScalarVector tGradientX ("gradientx", tNumDofs);
  Kokkos::deep_copy(tGradientX, 0.0);
  return tGradientX;
}

template<typename PhysicsType>
Plato::ScalarVector
CriterionEvaluatorSolutionFunction<PhysicsType>::
gradientState(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle
) const
{
  const Plato::OrdinalType tTotalNumNodes = mSpatialModel.Mesh->NumNodes();
  const Plato::OrdinalType tNumDofs = mNumDofsPerNode * tTotalNumNodes;
  Plato::ScalarVector tGradientU ("gradient control", tNumDofs);
  Plato::ScalarVector tStates = aDatabase.vector("states");

  auto tNodeIds = mSpatialModel.Mesh->GetNodeSetNodes(mDomainName);
  auto tNumNodes = tNodeIds.size();

  auto tNormal = mNormal;
  auto tTargetSolutionVector = mTargetSolutionVector;
  auto tNumDofsPerNode = mNumDofsPerNode;
  auto tTargetMagnitude = mTargetMagnitude;
  auto tTargetSolution = mTargetSolution;

  switch (mSolutionType)
  {
    case solution_type_t::DIFF_BETWEEN_SOLUTION_VECTOR_AND_TARGET_VECTOR:
      Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumNodes),
      KOKKOS_LAMBDA(Plato::OrdinalType aNodeOrdinal)
      {
        auto tIndex = tNodeIds[aNodeOrdinal];
        Plato::Scalar ds(0.0);
        for(int iDof=0; iDof<tNumDofsPerNode; iDof++)
        {
          auto dv = tStates(tNumDofsPerNode*tIndex+iDof);
          ds += (dv-tTargetSolutionVector[iDof])*(dv-tTargetSolutionVector[iDof]);
        }
        ds = (ds > 0.0) ? sqrt(ds) : ds;
        for(int iDof=0; iDof<tNumDofsPerNode; iDof++)
        {
          auto dv = tStates(tNumDofsPerNode*tIndex+iDof);
          if( ds != 0.0 )
          {
            tGradientU(tNumDofsPerNode*tIndex+iDof) = (dv-tTargetSolutionVector[iDof]) / (ds*tNumNodes);
          }
          else
          {
            tGradientU(tNumDofsPerNode*tIndex+iDof) = 0.0;
          }
        }
      }, "gradient_u");
      break; 
    case solution_type_t::DIFF_BETWEEN_SOLUTION_MAG_IN_DIRECTION_AND_TARGET:
      Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumNodes),
      KOKKOS_LAMBDA(Plato::OrdinalType aNodeOrdinal)
      {
        auto tIndex = tNodeIds[aNodeOrdinal];
        Plato::Scalar ds(0.0);
        for(int iDof=0; iDof<tNumDofsPerNode; iDof++)
        {
          auto dv = tStates(tNumDofsPerNode*tIndex+iDof);
          ds += (tNormal[iDof]*dv) * (tNormal[iDof]*dv);
        }
        ds = (ds > 0.0) ? sqrt(ds) : ds;
        Plato::Scalar tSign = (ds < tTargetMagnitude) ? -1.0 : 1.0;
        for(int iDof=0; iDof<tNumDofsPerNode; iDof++)
        {
          auto dv = tStates(tNumDofsPerNode*tIndex+iDof);
          if( ds != 0.0 )
          {
            tGradientU(tNumDofsPerNode*tIndex+iDof) = tSign * (tNormal[iDof] * (tNormal[iDof] * dv) / (ds*tNumNodes));
          }
          else
          {
            tGradientU(tNumDofsPerNode*tIndex+iDof) = 0.0;
          }
        }
      }, "gradient_u");
      break;
    case solution_type_t::SOLUTION_MAG_IN_DIRECTION:
      Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumNodes),
      KOKKOS_LAMBDA(Plato::OrdinalType aNodeOrdinal)
      {
        auto tIndex = tNodeIds[aNodeOrdinal];
        Plato::Scalar ds(0.0);
        for(int iDof=0; iDof<tNumDofsPerNode; iDof++)
        {
          auto dv = tStates(tNumDofsPerNode*tIndex+iDof);
          ds += (tNormal[iDof]*dv) * (tNormal[iDof]*dv);
        }
        ds = (ds > 0.0) ? sqrt(ds) : ds;
        for(int iDof=0; iDof<tNumDofsPerNode; iDof++)
        {
          auto dv = tStates(tNumDofsPerNode*tIndex+iDof);
          if( ds != 0.0 )
          {
            tGradientU(tNumDofsPerNode*tIndex+iDof) = tNormal[iDof] * (tNormal[iDof] * dv) / (ds*tNumNodes);
          }
          else
          {
            tGradientU(tNumDofsPerNode*tIndex+iDof) = 0.0;
          }
        }
      }, "gradient_u");
      break;
    case solution_type_t::SOLUTION_IN_DIRECTION:
      Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumNodes),
      KOKKOS_LAMBDA(Plato::OrdinalType aNodeOrdinal)
      {
        auto tIndex = tNodeIds[aNodeOrdinal];
        for(int iDof=0; iDof<tNumDofsPerNode; iDof++)
        {
          tGradientU(tNumDofsPerNode*tIndex+iDof) = tNormal[iDof] / tNumNodes;
        }
      }, "gradient_u");
      break; 
    case solution_type_t::DIFF_BETWEEN_SOLUTION_IN_DIRECTION_AND_TARGET_SOLUTION_IN_DIRECTION:
      Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumNodes),
      KOKKOS_LAMBDA(Plato::OrdinalType aNodeOrdinal)
      {
        Plato::Scalar tLocalValue(0.0);
        auto tIndex = tNodeIds[aNodeOrdinal];
        for(int iDof=0; iDof<tNumDofsPerNode; iDof++)
        {
          auto dv = tStates(tNumDofsPerNode*tIndex+iDof);
          tLocalValue += (tNormal[iDof]*dv);
        }
        Plato::Scalar tSign = (tLocalValue < tTargetSolution) ? -1.0 : 1.0;
        for(int iDof=0; iDof<tNumDofsPerNode; iDof++)
        {
          tGradientU(tNumDofsPerNode*tIndex+iDof) = tSign * tNormal[iDof] / tNumNodes;
        }
      }, "gradient_u");
      break; 
  }
  return tGradientU;
}

template<typename PhysicsType>
Plato::ScalarVector
CriterionEvaluatorSolutionFunction<PhysicsType>::
gradientNodeState(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle
) const
{
  const Plato::OrdinalType tNumNodes = mSpatialModel.Mesh->NumNodes();
  Plato::ScalarVector tGradientN ("gradient_n", tNumNodes);
  Kokkos::deep_copy(tGradientN, 0.0);
  return tGradientN;
}

template<typename PhysicsType>
Plato::ScalarVector
CriterionEvaluatorSolutionFunction<PhysicsType>::
gradientControl(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle
) const
{
  const Plato::OrdinalType tNumNodes = mSpatialModel.Mesh->NumNodes();
  Plato::ScalarVector tGradientZ ("gradientz", tNumNodes);
  Kokkos::deep_copy(tGradientZ, 0.0);
  return tGradientZ;
}

template<typename PhysicsType>
void
CriterionEvaluatorSolutionFunction<PhysicsType>::
updateProblem(
  const Plato::Database & aDatabase,
  const Plato::Scalar   & aCycle
) const 
{ return; }

template<typename PhysicsType>
void
CriterionEvaluatorSolutionFunction<PhysicsType>::
setFunctionName(const std::string aFunctionName)
{
  mFunctionName = aFunctionName;
}

template<typename PhysicsType>
std::string
CriterionEvaluatorSolutionFunction<PhysicsType>::
name() const
{
  return mFunctionName;
}

} // namespace Elliptic

} // namespace Plato
