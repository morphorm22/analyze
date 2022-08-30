#pragma once

#include "PlatoMathTypes.hpp"
#include "PlatoStaticsTypes.hpp"
#include "geometric/WorksetBase.hpp"
#include "geometric/LeastSquaresFunction.hpp"
#include "geometric/GeometryScalarFunction.hpp"

namespace Plato
{

namespace Geometric
{

/******************************************************************************//**
 * \brief Mass properties function class
 **********************************************************************************/
template<typename PhysicsType>
class MassPropertiesFunction :
    public Plato::Geometric::ScalarFunctionBase,
    public Plato::Geometric::WorksetBase<typename PhysicsType::ElementType>
{
private:
    using ElementType = typename PhysicsType::ElementType;

    using Residual  = typename Plato::Geometric::Evaluation<ElementType>::Residual;
    using GradientX = typename Plato::Geometric::Evaluation<ElementType>::GradientX;
    using GradientZ = typename Plato::Geometric::Evaluation<ElementType>::GradientZ;

    std::shared_ptr<Plato::Geometric::LeastSquaresFunction<PhysicsType>> mLeastSquaresFunction;

    const Plato::SpatialModel & mSpatialModel;

    Plato::DataMap& mDataMap; /*!< PLATO Engine and Analyze data map */

    std::string mFunctionName; /*!< User defined function name */

    std::map<std::string, Plato::Scalar> mMaterialDensities; /*!< material density */

    Plato::Matrix<3,3> mInertiaRotationMatrix;
    Plato::Array<3>    mInertiaPrincipalValues;
    Plato::Matrix<3,3> mMinusRotatedParallelAxisTheoremMatrix;

    Plato::Scalar mMeshExtentX;
    Plato::Scalar mMeshExtentY;
    Plato::Scalar mMeshExtentZ;

    /******************************************************************************//**
     * \brief Initialization of Mass Properties Function
     * \param [in] aMesh mesh database
     * \param [in] aProblemParams input parameters database
    **********************************************************************************/
    void initialize(Teuchos::ParameterList & aProblemParams);

    /******************************************************************************//**
     * \brief Create the least squares mass properties function
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aProblemParams input parameters database
    **********************************************************************************/
    void
    createLeastSquaresFunction(
        const Plato::SpatialModel    & aSpatialModel,
              Teuchos::ParameterList & aProblemParams
    );

    /******************************************************************************//**
     * \brief Check if all properties were specified by user
     * \param [in] aPropertyNames names of properties specified by user 
     * \return bool indicating if all properties were specified by user
    **********************************************************************************/
    bool allPropertiesSpecified(const std::vector<std::string>& aPropertyNames);


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
    void computeRotationAndParallelAxisTheoremMatrices(std::map<std::string, Plato::Scalar>& aGoldValueMap);

    /******************************************************************************//**
     * \brief Create an itemized least squares function for user specified mass properties
     * \param [in] aMesh mesh database
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
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \return physics scalar function
    **********************************************************************************/
    std::shared_ptr<Plato::Geometric::GeometryScalarFunction<PhysicsType>>
    getMassFunction(const Plato::SpatialModel & aSpatialModel);

    /******************************************************************************//**
     * \brief Create the 'first mass moment divided by the mass' function (CG)
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aMomentType mass moment type (FirstX, FirstY, FirstZ)
     * \return scalar function base
    **********************************************************************************/
    std::shared_ptr<Plato::Geometric::ScalarFunctionBase>
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
    std::shared_ptr<Plato::Geometric::ScalarFunctionBase>
    getSecondMassMoment(
        const Plato::SpatialModel & aSpatialModel,
        const std::string         & aMomentType
    );

    /******************************************************************************//**
     * \brief Create the moment of inertia function
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aAxes axes about which to compute the moment of inertia (XX, YY, ..)
     * \return scalar function base
    **********************************************************************************/
    std::shared_ptr<Plato::Geometric::ScalarFunctionBase>
    getMomentOfInertia(
        const Plato::SpatialModel & aSpatialModel,
        const std::string         & aAxes
    );

    /******************************************************************************//**
     * \brief Create the moment of inertia function about the CG in the principal coordinate frame
     * \param [in] aAxes axes about which to compute the moment of inertia (XX, YY, ..)
     * \return scalar function base
    **********************************************************************************/
    std::shared_ptr<Plato::Geometric::ScalarFunctionBase>
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
    getInertiaAndMassWeights(
              std::vector<Plato::Scalar> & aInertiaWeights, 
              Plato::Scalar              & aMassWeight, 
        const std::string                & aAxes
    );

public:
    /******************************************************************************//**
     * \brief Primary Mass Properties Function constructor
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato Analyze data map
     * \param [in] aProblemParams input parameters database
     * \param [in] aName user defined function name
    **********************************************************************************/
    MassPropertiesFunction(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              std::string            & aName
    );

    /******************************************************************************//**
     * \brief Compute the X, Y, and Z extents of the mesh (e.g. (X_max - X_min))
     * \param [in] aMesh mesh database
    **********************************************************************************/
    void computeMeshExtent(Plato::Mesh aMesh);

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aControl 1D view of control variables
     **********************************************************************************/
    void updateProblem(const Plato::ScalarVector & aControl) const override;

    /******************************************************************************//**
     * \brief Evaluate Mass Properties Function
     * \param [in] aControl 1D view of control variables
     * \return scalar function evaluation
    **********************************************************************************/
    Plato::Scalar value(const Plato::ScalarVector & aControl) const override;

    /******************************************************************************//**
     * \brief Evaluate gradient of the Mass Properties Function with respect to (wrt) the configuration parameters
     * \param [in] aControl 1D view of control variables
     * \return 1D view with the gradient of the scalar function wrt the configuration parameters
    **********************************************************************************/
    Plato::ScalarVector gradient_x(const Plato::ScalarVector & aControl) const override;

    /******************************************************************************//**
     * \brief Evaluate gradient of the Mass Properties Function with respect to (wrt) the control variables
     * \param [in] aControl 1D view of control variables
     * \return 1D view with the gradient of the scalar function wrt the control variables
    **********************************************************************************/
    Plato::ScalarVector gradient_z(const Plato::ScalarVector & aControl) const override;

    /******************************************************************************//**
     * \brief Return user defined function name
     * \return User defined function name
    **********************************************************************************/
    std::string name() const;
};
// class MassPropertiesFunction

} // namespace Geometric

} // namespace Plato
