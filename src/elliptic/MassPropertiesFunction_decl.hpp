#pragma once

#include "WorksetBase.hpp"
#include "elliptic/LeastSquaresFunction.hpp"
#include "elliptic/PhysicsScalarFunction.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Mass properties function class
 **********************************************************************************/
template<typename PhysicsType>
class MassPropertiesFunction :
    public Plato::Elliptic::ScalarFunctionBase,
    public Plato::WorksetBase<typename PhysicsType::ElementType>
{
private:
    using ElementType = typename PhysicsType::ElementType;

    using Residual  = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    using GradientU = typename Plato::Elliptic::Evaluation<ElementType>::Jacobian;
    using GradientX = typename Plato::Elliptic::Evaluation<ElementType>::GradientX;
    using GradientZ = typename Plato::Elliptic::Evaluation<ElementType>::GradientZ;

    std::shared_ptr<Plato::Elliptic::LeastSquaresFunction<PhysicsType>> mLeastSquaresFunction;

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
    computeRotationAndParallelAxisTheoremMatrices(std::map<std::string, Plato::Scalar>& aGoldValueMap);

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
    std::shared_ptr<Plato::Elliptic::PhysicsScalarFunction<PhysicsType>>
    getMassFunction(
        const Plato::SpatialModel & aSpatialModel
    );

    /******************************************************************************//**
     * \brief Create the 'first mass moment divided by the mass' function (CG)
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aMomentType mass moment type (FirstX, FirstY, FirstZ)
     * \return scalar function base
    **********************************************************************************/
    std::shared_ptr<Plato::Elliptic::ScalarFunctionBase>
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
    std::shared_ptr<Plato::Elliptic::ScalarFunctionBase>
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
    std::shared_ptr<Plato::Elliptic::ScalarFunctionBase>
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
    std::shared_ptr<Plato::Elliptic::ScalarFunctionBase>
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
    MassPropertiesFunction(
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

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aState 1D view of state variables
     * \param [in] aControl 1D view of control variables
     **********************************************************************************/
    void
    updateProblem(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aControl
    ) const override;

    /******************************************************************************//**
     * \brief Evaluate Mass Properties Function
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return scalar function evaluation
    **********************************************************************************/
    Plato::Scalar
    value(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const override;

    /******************************************************************************//**
     * \brief Evaluate gradient of the Mass Properties Function with respect to (wrt) the state variables
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_u(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::OrdinalType    aStepIndex,
              Plato::Scalar         aTimeStep = 0.0
    ) const override;

    /******************************************************************************//**
     * \brief Evaluate gradient of the Mass Properties Function with respect to (wrt) the configuration
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_x(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const override;

    /******************************************************************************//**
     * \brief Evaluate gradient of the Mass Properties Function with respect to (wrt) the control
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_z(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const override;

    /******************************************************************************//**
     * \brief Return user defined function name
     * \return User defined function name
    **********************************************************************************/
    std::string name() const;
};
// class MassPropertiesFunction

} // namespace Elliptic

} // namespace Plato
