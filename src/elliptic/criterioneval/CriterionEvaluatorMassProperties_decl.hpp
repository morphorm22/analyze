#pragma once

#include "base/WorksetBase.hpp"
#include "elliptic/criterioneval/CriterionEvaluatorLeastSquares.hpp"
#include "elliptic/criterioneval/CriterionEvaluatorScalarFunction.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Mass properties function class
 **********************************************************************************/
template<typename PhysicsType>
class CriterionEvaluatorMassProperties :
    public Plato::Elliptic::CriterionEvaluatorBase
{
private:
  /// @brief local topological element typename
  using ElementType = typename PhysicsType::ElementType;  
  using Residual  = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
  using GradientU = typename Plato::Elliptic::Evaluation<ElementType>::Jacobian;
  using GradientX = typename Plato::Elliptic::Evaluation<ElementType>::GradientX;
  using GradientZ = typename Plato::Elliptic::Evaluation<ElementType>::GradientZ;  
  /// @brief least square function evaluiator
  std::shared_ptr<Plato::Elliptic::CriterionEvaluatorLeastSquares<PhysicsType>> mLeastSquaresFunction;  
  /// @brief contains mesh and model information
  const Plato::SpatialModel & mSpatialModel;  
  /// @brief output database
  Plato::DataMap& mDataMap;
  /// @brief evaluator name 
  std::string mFunctionName; 
  /// @brief map from domain to material density
  std::map<std::string, Plato::Scalar> mMaterialDensities;
  Plato::Matrix<3,3> mInertiaRotationMatrix;
  Plato::Array<3>    mInertiaPrincipalValues;
  Plato::Matrix<3,3> mMinusRotatedParallelAxisTheoremMatrix;  
  Plato::Scalar mMeshExtentX;
  Plato::Scalar mMeshExtentY;
  Plato::Scalar mMeshExtentZ;

private:
  /******************************************************************************//**
   * \brief Initialization of Mass Properties Function
   * \param [in] aInputParams input parameters database
  **********************************************************************************/
  void
  initialize(
      Teuchos::ParameterList & aInputParams
  );  
  /******************************************************************************//**
   * \brief Create the least squares mass properties function
   * \param [in] aSpatialModel Plato Analyze spatial model
   * \param [in] aInputParams input parameters database
  **********************************************************************************/
  void
  createLeastSquaresFunction(
      const Plato::SpatialModel    & aSpatialModel,
            Teuchos::ParameterList & aInputParams
  );  
  /******************************************************************************//**
   * \brief Check if all properties were specified by user
   * \param [in] aPropertyNames names of properties specified by user 
   * \return bool indicating if all properties were specified by user
  **********************************************************************************/
  bool
  allPropertiesSpecified(const std::vector<std::string>& aPropertyNames);  
  /******************************************************************************//**
   * \brief Create a least squares function for all mass properties (inertia about gold CG)
   * \param [in] aSpatialModel Plato Analyze spatial model
   * \param [in] aPropertyNames names of properties specified by user 
   * \param [in] aPropertyWeights weights of properties specified by user 
   * \param [in] aPropertyGoldValues gold values of properties specified by user 
  **********************************************************************************/
  void
  createAllMassPropertiesLeastSquaresFunction(
      const Plato::SpatialModel        & aSpatialModel,
      const std::vector<std::string>   & aPropertyNames,
      const std::vector<Plato::Scalar> & aPropertyWeights,
      const std::vector<Plato::Scalar> & aPropertyGoldValues
  );  
  /******************************************************************************//**
   * \brief Compute rotation and parallel axis theorem matrices
   * \param [in] aGoldValueMap gold value map
  **********************************************************************************/
  void
  computeRotationAndParallelAxisTheoremMatrices(
      std::map<std::string, Plato::Scalar>& aGoldValueMap
  );  
  /******************************************************************************//**
   * \brief Create an itemized least squares function for user specified mass properties
   * \param [in] aPropertyNames names of properties specified by user 
   * \param [in] aPropertyWeights weights of properties specified by user 
   * \param [in] aPropertyGoldValues gold values of properties specified by user 
  **********************************************************************************/
  void
  createItemizedLeastSquaresFunction(
      const Plato::SpatialModel        & aSpatialModel,
      const std::vector<std::string>   & aPropertyNames,
      const std::vector<Plato::Scalar> & aPropertyWeights,
      const std::vector<Plato::Scalar> & aPropertyGoldValues
  );  
  /******************************************************************************//**
   * \brief Create the mass function only
   * \return physics scalar function
  **********************************************************************************/
  std::shared_ptr<Plato::Elliptic::CriterionEvaluatorScalarFunction<PhysicsType>>
  getMassFunction(
      const Plato::SpatialModel & aSpatialModel
  );  
  /******************************************************************************//**
   * \brief Create the 'first mass moment divided by the mass' function (CG)
   * \param [in] aSpatialModel Plato Analyze spatial model
   * \param [in] aMomentType mass moment type (FirstX, FirstY, FirstZ)
   * \return scalar function base
  **********************************************************************************/
  std::shared_ptr<Plato::Elliptic::CriterionEvaluatorBase>
  getFirstMomentOverMassRatio(
      const Plato::SpatialModel & aSpatialModel,
      const std::string         & aMomentType
  );  
  /******************************************************************************//**
   * \brief Create the second mass moment function
   * \param [in] aSpatialModel Plato Analyze spatial model
   * \param [in] aMomentType second mass moment type (XX, XY, YY, ...)
   * \return scalar function base
  **********************************************************************************/
  std::shared_ptr<Plato::Elliptic::CriterionEvaluatorBase>
  getSecondMassMoment(
      const Plato::SpatialModel & aSpatialModel,
      const std::string & aMomentType
  );  
  /******************************************************************************//**
   * \brief Create the moment of inertia function
   * \param [in] aSpatialModel Plato Analyze spatial domain
   * \param [in] aAxes axes about which to compute the moment of inertia (XX, YY, ..)
   * \return scalar function base
  **********************************************************************************/
  std::shared_ptr<Plato::Elliptic::CriterionEvaluatorBase>
  getMomentOfInertia(
      const Plato::SpatialModel & aSpatialModel,
      const std::string & aAxes
  );  
  /******************************************************************************//**
   * \brief Create the moment of inertia function about the CG in the principal coordinate frame
   * \param [in] aSpatialModel Plato Analyze spatial model
   * \param [in] aAxes axes about which to compute the moment of inertia (XX, YY, ..)
   * \return scalar function base
  **********************************************************************************/
  std::shared_ptr<Plato::Elliptic::CriterionEvaluatorBase>
  getMomentOfInertiaRotatedAboutCG(
      const Plato::SpatialModel & aSpatialModel,
      const std::string         & aAxes
  );  
  /******************************************************************************//**
   * \brief Compute the inertia weights and mass weight for the inertia about the CG rotated into principal frame
   * \param [out] aInertiaWeights inertia weights
   * \param [out] aMassWeight mass weight
   * \param [in] aAxes axes about which to compute the moment of inertia (XX, YY, ..)
  **********************************************************************************/
  void
  getInertiaAndMassWeights(std::vector<Plato::Scalar> & aInertiaWeights, 
                           Plato::Scalar & aMassWeight, 
                           const std::string & aAxes);

public:
  /******************************************************************************//**
   * \brief Primary Mass Properties Function constructor
   * \param [in] aSpatialModel Plato Analyze spatial model
   * \param [in] aDataMap PLATO Engine and Analyze data map
   * \param [in] aInputParams input parameters database
   * \param [in] aName user defined function name
  **********************************************************************************/
  CriterionEvaluatorMassProperties(
      const Plato::SpatialModel    & aSpatialModel,
            Plato::DataMap         & aDataMap,
            Teuchos::ParameterList & aInputParams,
            std::string            & aName
  );  
  /******************************************************************************//**
   * \brief Compute the X, Y, and Z extents of the mesh (e.g. (X_max - X_min))
   * \param [in] aMesh mesh database
  **********************************************************************************/
  void
  computeMeshExtent(Plato::Mesh aMesh);  

  /// @fn isLinear
  /// @brief return true if scalar function is linear
  /// @return boolean
  bool 
  isLinear() 
  const;

  /// @fn updateProblem
  /// @brief update criterion parameters at runtime
  /// @param [in] aDatabase function domain and range database
  /// @param [in] aCycle    scalar, e.g.; time step
  void
  updateProblem(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle
  ) const override;  

  /// @fn value
  /// @brief evaluate mass properties function
  /// @param [in] aDatabase function domain and range database
  /// @param [in] aCycle    scalar, e.g.; time step
  /// @return scalar
  Plato::Scalar
  value(const Plato::Database & aDatabase,
        const Plato::Scalar   & aCycle
  ) const;  

  /// @fn gradientState
  /// @brief compute partial derivative of mass properties function with respect to states
  /// @param [in] aDatabase function domain and range database
  /// @param [in] aCycle    scalar, e.g.; time step
  /// @return plato scalar vector
  Plato::ScalarVector
  gradientState(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle
  ) const;  

  /// @fn gradientConfig
  /// @brief compute partial derivative of mass properties function with respect to configuration
  /// @param [in] aDatabase function domain and range database
  /// @param [in] aCycle    scalar, e.g.; time step
  /// @return plato scalar vector
  Plato::ScalarVector
  gradientConfig(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle
  ) const;

  /// @fn gradientControl
  /// @brief compute partial derivative of mass properties function with respect to controls
  /// @param [in] aDatabase function domain and range database
  /// @param [in] aCycle    scalar, e.g.; time step
  /// @return plato scalar vector
  Plato::ScalarVector
  gradientControl(
    const Plato::Database & aDatabase,
    const Plato::Scalar   & aCycle
  ) const;  

  /// @fn name
  /// @brief return criterion evaluator name
  /// @return string
  std::string 
  name() 
  const;
};
// class CriterionEvaluatorMassProperties

} // namespace Elliptic

} // namespace Plato
