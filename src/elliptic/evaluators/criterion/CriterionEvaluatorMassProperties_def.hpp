#pragma once

#include <set>

#include "BLAS1.hpp"
#include "PlatoEigen.hpp"
#include "AnalyzeMacros.hpp"
#include "PlatoStaticsTypes.hpp"
#include "elliptic/mechanical/linear/CriterionMassMoment.hpp"
#include "elliptic/evaluators/criterion/CriterionEvaluatorDivision.hpp"
#include "elliptic/evaluators/criterion/CriterionEvaluatorWeightedSum.hpp"
#include "elliptic/evaluators/criterion/CriterionEvaluatorLeastSquares.hpp"
#include "elliptic/evaluators/criterion/FactoryCriterionEvaluator.hpp"

namespace Plato
{

namespace Elliptic
{

    /******************************************************************************//**
     * \brief Initialization of Mass Properties Function
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    template<typename PhysicsType>
    void
    CriterionEvaluatorMassProperties<PhysicsType>::initialize(
        Teuchos::ParameterList & aInputParams
    )
    {
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();

            auto tMaterialModelsInputs = aInputParams.get<Teuchos::ParameterList>("Material Models");
            if( tMaterialModelsInputs.isSublist(tDomain.getMaterialName()) )
            {
                auto tMaterialModelInputs = aInputParams.sublist(tDomain.getMaterialName());
                mMaterialDensities[tName] = tMaterialModelInputs.get<Plato::Scalar>("Density", 1.0);
            }
        }
        createLeastSquaresFunction(mSpatialModel, aInputParams);
    }

    /******************************************************************************//**
     * \brief Create the least squares mass properties function
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    template<typename PhysicsType>
    void
    CriterionEvaluatorMassProperties<PhysicsType>::createLeastSquaresFunction(
        const Plato::SpatialModel    & aSpatialModel,
              Teuchos::ParameterList & aInputParams
    )
    {
        auto tProblemFunctionName = aInputParams.sublist("Criteria").sublist(mFunctionName);

        auto tPropertyNamesTeuchos      = tProblemFunctionName.get<Teuchos::Array<std::string>>("Properties");
        auto tPropertyWeightsTeuchos    = tProblemFunctionName.get<Teuchos::Array<Plato::Scalar>>("Weights");
        auto tPropertyGoldValuesTeuchos = tProblemFunctionName.get<Teuchos::Array<Plato::Scalar>>("Gold Values");

        auto tPropertyNames      = tPropertyNamesTeuchos.toVector();
        auto tPropertyWeights    = tPropertyWeightsTeuchos.toVector();
        auto tPropertyGoldValues = tPropertyGoldValuesTeuchos.toVector();

        if (tPropertyNames.size() != tPropertyWeights.size())
        {
            const std::string tErrorString = std::string("Number of 'Properties' in '") + mFunctionName + 
                                                         "' parameter list does not equal the number of 'Weights'";
            ANALYZE_THROWERR(tErrorString)
        }

        if (tPropertyNames.size() != tPropertyGoldValues.size())
        {
            const std::string tErrorString = std::string("Number of 'Gold Values' in '") + mFunctionName + 
                                                         "' parameter list does not equal the number of 'Properties'";
            ANALYZE_THROWERR(tErrorString)
        }

        const bool tAllPropertiesSpecifiedByUser = allPropertiesSpecified(tPropertyNames);

        computeMeshExtent(aSpatialModel.Mesh);

        if (tAllPropertiesSpecifiedByUser)
            createAllMassPropertiesLeastSquaresFunction(
                aSpatialModel, tPropertyNames, tPropertyWeights, tPropertyGoldValues);
        else
            createItemizedLeastSquaresFunction(
                aSpatialModel, tPropertyNames, tPropertyWeights, tPropertyGoldValues);
    }

    /******************************************************************************//**
     * \brief Check if all properties were specified by user
     * \param [in] aPropertyNames names of properties specified by user 
     * \return bool indicating if all properties were specified by user
    **********************************************************************************/
    template<typename PhysicsType>
    bool
    CriterionEvaluatorMassProperties<PhysicsType>::allPropertiesSpecified(const std::vector<std::string>& aPropertyNames)
    {
        // copy the vector since we sort it and remove items in this function
        std::vector<std::string> tPropertyNames(aPropertyNames.begin(), aPropertyNames.end());

        const unsigned int tUserSpecifiedNumberOfProperties = tPropertyNames.size();

        // Sort and erase duplicate entries
        std::sort( tPropertyNames.begin(), tPropertyNames.end() );
        tPropertyNames.erase( std::unique( tPropertyNames.begin(), tPropertyNames.end() ), tPropertyNames.end());

        // Check for duplicate entries from the user
        const unsigned int tUniqueNumberOfProperties = tPropertyNames.size();
        if (tUserSpecifiedNumberOfProperties != tUniqueNumberOfProperties)
        { ANALYZE_THROWERR("User specified mass properties vector contains duplicate entries!") }

        if (tUserSpecifiedNumberOfProperties < 10) return false;

        std::vector<std::string> tAllPropertiesVector = 
                                 {"Mass","CGx","CGy","CGz","Ixx","Iyy","Izz","Ixy","Ixz","Iyz"};
        std::sort(tAllPropertiesVector.begin(), tAllPropertiesVector.end());
        
        std::set<std::string> tAllPropertiesSet(tAllPropertiesVector.begin(), tAllPropertiesVector.end());
        std::set<std::string>::iterator tSetIterator;

        // if number of unqiue user-specified properties does not equal all of them, return false
        if (tPropertyNames.size() != tAllPropertiesVector.size()) return false;

        for (Plato::OrdinalType tIndex = 0; tIndex < tPropertyNames.size(); ++tIndex)
        {
            const std::string tCurrentProperty = tPropertyNames[tIndex];

            // Check to make sure it is a valid property
            tSetIterator = tAllPropertiesSet.find(tCurrentProperty);
            if (tSetIterator == tAllPropertiesSet.end())
            {
                const std::string tErrorString = std::string("Specified mass property '") +
                tCurrentProperty + "' not implemented. Options are: Mass, CGx, CGy, CGz, " 
                                 + "Ixx, Iyy, Izz, Ixy, Ixz, Iyz";
                ANALYZE_THROWERR(tErrorString)
            }

            // property vectors were sorted so check that the properties match in sequence
            if (tCurrentProperty != tAllPropertiesVector[tIndex])
            {
                printf("Property %s does not equal property %s \n", 
                       tCurrentProperty.c_str(), tAllPropertiesVector[tIndex].c_str());
                printf("If user specifies all mass properties, better performance may be experienced.\n");
                return false;
            }
        }

        return true;
    }


    /******************************************************************************//**
     * \brief Create a least squares function for all mass properties (inertia about gold CG)
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aPropertyNames names of properties specified by user 
     * \param [in] aPropertyWeights weights of properties specified by user 
     * \param [in] aPropertyGoldValues gold values of properties specified by user 
    **********************************************************************************/
    template<typename PhysicsType>
    void
    CriterionEvaluatorMassProperties<PhysicsType>::createAllMassPropertiesLeastSquaresFunction(
        const Plato::SpatialModel        & aSpatialModel,
        const std::vector<std::string>   & aPropertyNames,
        const std::vector<Plato::Scalar> & aPropertyWeights,
        const std::vector<Plato::Scalar> & aPropertyGoldValues
    )
    {
        printf("Creating all mass properties function.\n");
        mLeastSquaresFunction = 
          std::make_shared<Plato::Elliptic::CriterionEvaluatorLeastSquares<PhysicsType>>(aSpatialModel, mDataMap);
        std::map<std::string, Plato::Scalar> tWeightMap;
        std::map<std::string, Plato::Scalar> tGoldValueMap;
        for (Plato::OrdinalType tPropertyIndex = 0; tPropertyIndex < aPropertyNames.size(); ++tPropertyIndex)
        {
            const std::string   tPropertyName      = aPropertyNames[tPropertyIndex];
            const Plato::Scalar tPropertyWeight    = aPropertyWeights[tPropertyIndex];
            const Plato::Scalar tPropertyGoldValue = aPropertyGoldValues[tPropertyIndex];

            tWeightMap.insert(    std::pair<std::string, Plato::Scalar>(tPropertyName, tPropertyWeight   ) );
            tGoldValueMap.insert( std::pair<std::string, Plato::Scalar>(tPropertyName, tPropertyGoldValue) );
        }

        computeRotationAndParallelAxisTheoremMatrices(tGoldValueMap);

        // Mass
        mLeastSquaresFunction->allocateScalarFunctionBase(getMassFunction(aSpatialModel));
        mLeastSquaresFunction->appendFunctionWeight(tWeightMap[std::string("Mass")]);
        mLeastSquaresFunction->appendGoldFunctionValue(tGoldValueMap[std::string("Mass")]);

        // CGx
        mLeastSquaresFunction->allocateScalarFunctionBase(getFirstMomentOverMassRatio(aSpatialModel, "FirstX"));
        mLeastSquaresFunction->appendFunctionWeight(tWeightMap[std::string("CGx")]);
        mLeastSquaresFunction->appendGoldFunctionValue(tGoldValueMap[std::string("CGx")], false);
        mLeastSquaresFunction->appendFunctionNormalization(mMeshExtentX);

        // CGy
        mLeastSquaresFunction->allocateScalarFunctionBase(getFirstMomentOverMassRatio(aSpatialModel, "FirstY"));
        mLeastSquaresFunction->appendFunctionWeight(tWeightMap[std::string("CGy")]);
        mLeastSquaresFunction->appendGoldFunctionValue(tGoldValueMap[std::string("CGy")], false);
        mLeastSquaresFunction->appendFunctionNormalization(mMeshExtentY);

        // CGz
        mLeastSquaresFunction->allocateScalarFunctionBase(getFirstMomentOverMassRatio(aSpatialModel, "FirstZ"));
        mLeastSquaresFunction->appendFunctionWeight(tWeightMap[std::string("CGz")]);
        mLeastSquaresFunction->appendGoldFunctionValue(tGoldValueMap[std::string("CGz")], false);
        mLeastSquaresFunction->appendFunctionNormalization(mMeshExtentZ);

        // Ixx
        mLeastSquaresFunction->allocateScalarFunctionBase(getMomentOfInertiaRotatedAboutCG(aSpatialModel, "XX"));
        mLeastSquaresFunction->appendFunctionWeight(tWeightMap[std::string("Ixx")]);
        mLeastSquaresFunction->appendGoldFunctionValue(mInertiaPrincipalValues(0));

        // Iyy
        mLeastSquaresFunction->allocateScalarFunctionBase(getMomentOfInertiaRotatedAboutCG(aSpatialModel, "YY"));
        mLeastSquaresFunction->appendFunctionWeight(tWeightMap[std::string("Iyy")]);
        mLeastSquaresFunction->appendGoldFunctionValue(mInertiaPrincipalValues(1));

        // Izz
        mLeastSquaresFunction->allocateScalarFunctionBase(getMomentOfInertiaRotatedAboutCG(aSpatialModel, "ZZ"));
        mLeastSquaresFunction->appendFunctionWeight(tWeightMap[std::string("Izz")]);
        mLeastSquaresFunction->appendGoldFunctionValue(mInertiaPrincipalValues(2));


        // Minimum Principal Moment of Inertia
        Plato::Scalar tMinPrincipalMoment = std::min(mInertiaPrincipalValues(0),
                                            std::min(mInertiaPrincipalValues(1), mInertiaPrincipalValues(2)));

        // Ixy
        mLeastSquaresFunction->allocateScalarFunctionBase(getMomentOfInertiaRotatedAboutCG(aSpatialModel, "XY"));
        mLeastSquaresFunction->appendFunctionWeight(tWeightMap[std::string("Ixy")]);
        mLeastSquaresFunction->appendGoldFunctionValue(0.0, false);
        mLeastSquaresFunction->appendFunctionNormalization(tMinPrincipalMoment);

        // Ixz
        mLeastSquaresFunction->allocateScalarFunctionBase(getMomentOfInertiaRotatedAboutCG(aSpatialModel, "XZ"));
        mLeastSquaresFunction->appendFunctionWeight(tWeightMap[std::string("Ixz")]);
        mLeastSquaresFunction->appendGoldFunctionValue(0.0, false);
        mLeastSquaresFunction->appendFunctionNormalization(tMinPrincipalMoment);

        // Iyz
        mLeastSquaresFunction->allocateScalarFunctionBase(getMomentOfInertiaRotatedAboutCG(aSpatialModel, "YZ"));
        mLeastSquaresFunction->appendFunctionWeight(tWeightMap[std::string("Iyz")]);
        mLeastSquaresFunction->appendGoldFunctionValue(0.0, false);
        mLeastSquaresFunction->appendFunctionNormalization(tMinPrincipalMoment);
    }

    /******************************************************************************//**
     * \brief Compute rotation and parallel axis theorem matrices
     * \param [in] aGoldValueMap gold value map
    **********************************************************************************/
    template<typename PhysicsType>
    void
    CriterionEvaluatorMassProperties<PhysicsType>::computeRotationAndParallelAxisTheoremMatrices(std::map<std::string, Plato::Scalar>& aGoldValueMap)
    {
        const Plato::Scalar Mass = aGoldValueMap[std::string("Mass")];

        const Plato::Scalar Ixx = aGoldValueMap[std::string("Ixx")];
        const Plato::Scalar Iyy = aGoldValueMap[std::string("Iyy")];
        const Plato::Scalar Izz = aGoldValueMap[std::string("Izz")];
        const Plato::Scalar Ixy = aGoldValueMap[std::string("Ixy")];
        const Plato::Scalar Ixz = aGoldValueMap[std::string("Ixz")];
        const Plato::Scalar Iyz = aGoldValueMap[std::string("Iyz")];

        const Plato::Scalar CGx = aGoldValueMap[std::string("CGx")];
        const Plato::Scalar CGy = aGoldValueMap[std::string("CGy")]; 
        const Plato::Scalar CGz = aGoldValueMap[std::string("CGz")];

        Plato::Array<3> tCGVector({CGx, CGy, CGz});

        const Plato::Scalar tNormSquared = Plato::dot(tCGVector, tCGVector);

        Plato::Matrix<3,3> tParallelAxisTheoremMatrix = Plato::plus(Plato::identity<3>(tNormSquared), Plato::outer_product(tCGVector, tCGVector), -1.0);
        Plato::Matrix<3,3> tGoldInertiaTensor({Ixx,Ixy,Ixz, Ixy,Iyy,Iyz, Ixz,Iyz,Izz});

        Plato::Matrix<3,3> tGoldInertiaTensorAboutCG = Plato::plus(tGoldInertiaTensor, tParallelAxisTheoremMatrix, -Mass);
    
        Plato::decomposeEigenJacobi<3>(tGoldInertiaTensorAboutCG, mInertiaRotationMatrix, mInertiaPrincipalValues);

        printf("Eigenvalues of GoldInertiaTensor : %f, %f, %f\n", mInertiaPrincipalValues(0), 
            mInertiaPrincipalValues(1), mInertiaPrincipalValues(2));

        mMinusRotatedParallelAxisTheoremMatrix = Plato::times(-1.0,
            Plato::times(Plato::transpose(mInertiaRotationMatrix), Plato::times(tParallelAxisTheoremMatrix, mInertiaRotationMatrix)));
    }

    /******************************************************************************//**
     * \brief Create an itemized least squares function for user specified mass properties
     * \param [in] aPropertyNames names of properties specified by user 
     * \param [in] aPropertyWeights weights of properties specified by user 
     * \param [in] aPropertyGoldValues gold values of properties specified by user 
    **********************************************************************************/
    template<typename PhysicsType>
    void
    CriterionEvaluatorMassProperties<PhysicsType>::createItemizedLeastSquaresFunction(
        const Plato::SpatialModel        & aSpatialModel,
        const std::vector<std::string>   & aPropertyNames,
        const std::vector<Plato::Scalar> & aPropertyWeights,
        const std::vector<Plato::Scalar> & aPropertyGoldValues
    )
    {
        printf("Creating itemized mass properties function.\n");
        mLeastSquaresFunction = 
          std::make_shared<Plato::Elliptic::CriterionEvaluatorLeastSquares<PhysicsType>>(aSpatialModel, mDataMap);
        for (Plato::OrdinalType tPropertyIndex = 0; tPropertyIndex < aPropertyNames.size(); ++tPropertyIndex)
        {
            const std::string   tPropertyName      = aPropertyNames[tPropertyIndex];
            const Plato::Scalar tPropertyWeight    = aPropertyWeights[tPropertyIndex];
            const Plato::Scalar tPropertyGoldValue = aPropertyGoldValues[tPropertyIndex];

            if (tPropertyName == "Mass")
            {
                mLeastSquaresFunction->allocateScalarFunctionBase(getMassFunction(aSpatialModel));
                mLeastSquaresFunction->appendFunctionWeight(tPropertyWeight);
                mLeastSquaresFunction->appendGoldFunctionValue(tPropertyGoldValue);
            }
            else if (tPropertyName == "CGx")
            {
                mLeastSquaresFunction->allocateScalarFunctionBase(getFirstMomentOverMassRatio(aSpatialModel, "FirstX"));
                mLeastSquaresFunction->appendFunctionWeight(tPropertyWeight);
                mLeastSquaresFunction->appendGoldFunctionValue(tPropertyGoldValue, false);
                mLeastSquaresFunction->appendFunctionNormalization(mMeshExtentX);
            }
            else if (tPropertyName == "CGy")
            {
                mLeastSquaresFunction->allocateScalarFunctionBase(getFirstMomentOverMassRatio(aSpatialModel, "FirstY"));
                mLeastSquaresFunction->appendFunctionWeight(tPropertyWeight);
                mLeastSquaresFunction->appendGoldFunctionValue(tPropertyGoldValue, false);
                mLeastSquaresFunction->appendFunctionNormalization(mMeshExtentY);
            }
            else if (tPropertyName == "CGz")
            {
                mLeastSquaresFunction->allocateScalarFunctionBase(getFirstMomentOverMassRatio(aSpatialModel, "FirstZ"));
                mLeastSquaresFunction->appendFunctionWeight(tPropertyWeight);
                mLeastSquaresFunction->appendGoldFunctionValue(tPropertyGoldValue, false);
                mLeastSquaresFunction->appendFunctionNormalization(mMeshExtentZ);
            }
            else if (tPropertyName == "Ixx")
            {
                mLeastSquaresFunction->allocateScalarFunctionBase(getMomentOfInertia(aSpatialModel, "XX"));
                mLeastSquaresFunction->appendFunctionWeight(tPropertyWeight);
                mLeastSquaresFunction->appendGoldFunctionValue(tPropertyGoldValue);
            }
            else if (tPropertyName == "Iyy")
            {
                mLeastSquaresFunction->allocateScalarFunctionBase(getMomentOfInertia(aSpatialModel, "YY"));
                mLeastSquaresFunction->appendFunctionWeight(tPropertyWeight);
                mLeastSquaresFunction->appendGoldFunctionValue(tPropertyGoldValue);
            }
            else if (tPropertyName == "Izz")
            {
                mLeastSquaresFunction->allocateScalarFunctionBase(getMomentOfInertia(aSpatialModel, "ZZ"));
                mLeastSquaresFunction->appendFunctionWeight(tPropertyWeight);
                mLeastSquaresFunction->appendGoldFunctionValue(tPropertyGoldValue);
            }
            else if (tPropertyName == "Ixy")
            {
                mLeastSquaresFunction->allocateScalarFunctionBase(getMomentOfInertia(aSpatialModel, "XY"));
                mLeastSquaresFunction->appendFunctionWeight(tPropertyWeight);
                mLeastSquaresFunction->appendGoldFunctionValue(tPropertyGoldValue);
            }
            else if (tPropertyName == "Ixz")
            {
                mLeastSquaresFunction->allocateScalarFunctionBase(getMomentOfInertia(aSpatialModel, "XZ"));
                mLeastSquaresFunction->appendFunctionWeight(tPropertyWeight);
                mLeastSquaresFunction->appendGoldFunctionValue(tPropertyGoldValue);
            }
            else if (tPropertyName == "Iyz")
            {
                mLeastSquaresFunction->allocateScalarFunctionBase(getMomentOfInertia(aSpatialModel, "YZ"));
                mLeastSquaresFunction->appendFunctionWeight(tPropertyWeight);
                mLeastSquaresFunction->appendGoldFunctionValue(tPropertyGoldValue);
            }
            else
            {
                const std::string tErrorString = std::string("Specified mass property '") +
                tPropertyName + "' not implemented. Options are: Mass, CGx, CGy, CGz, " 
                              + "Ixx, Iyy, Izz, Ixy, Ixz, Iyz";
                ANALYZE_THROWERR(tErrorString)
            }
        }
    }

    /******************************************************************************//**
     * \brief Create the mass function only
     * \return physics scalar function
    **********************************************************************************/
    template<typename PhysicsType>
    std::shared_ptr<Plato::Elliptic::CriterionEvaluatorScalarFunction<PhysicsType>>
    CriterionEvaluatorMassProperties<PhysicsType>::getMassFunction(
        const Plato::SpatialModel & aSpatialModel
    )
    {
        std::shared_ptr<Plato::Elliptic::CriterionEvaluatorScalarFunction<PhysicsType>> tMassFunction =
          std::make_shared<Plato::Elliptic::CriterionEvaluatorScalarFunction<PhysicsType>>(aSpatialModel, mDataMap);
        tMassFunction->setFunctionName("Mass Function");

        std::string tCalculationType = std::string("Mass");

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();

            std::shared_ptr<Plato::Elliptic::CriterionMassMoment<Residual>> tValue = 
                 std::make_shared<Plato::Elliptic::CriterionMassMoment<Residual>>(tDomain, mDataMap);
            tValue->setMaterialDensity(mMaterialDensities[tName]);
            tValue->setCalculationType(tCalculationType);
            tMassFunction->setEvaluator(Plato::Elliptic::evaluator_t::VALUE, tValue, tName);

            std::shared_ptr<Plato::Elliptic::CriterionMassMoment<GradientU>> tGradientU = 
                 std::make_shared<Plato::Elliptic::CriterionMassMoment<GradientU>>(tDomain, mDataMap);
            tGradientU->setMaterialDensity(mMaterialDensities[tName]);
            tGradientU->setCalculationType(tCalculationType);
            tMassFunction->setEvaluator(Plato::Elliptic::evaluator_t::GRAD_U, tGradientU, tName);

            std::shared_ptr<Plato::Elliptic::CriterionMassMoment<GradientZ>> tGradientZ = 
                 std::make_shared<Plato::Elliptic::CriterionMassMoment<GradientZ>>(tDomain, mDataMap);
            tGradientZ->setMaterialDensity(mMaterialDensities[tName]);
            tGradientZ->setCalculationType(tCalculationType);
            tMassFunction->setEvaluator(Plato::Elliptic::evaluator_t::GRAD_Z, tGradientZ, tName);

            std::shared_ptr<Plato::Elliptic::CriterionMassMoment<GradientX>> tGradientX = 
                 std::make_shared<Plato::Elliptic::CriterionMassMoment<GradientX>>(tDomain, mDataMap);
            tGradientX->setMaterialDensity(mMaterialDensities[tName]);
            tGradientX->setCalculationType(tCalculationType);
            tMassFunction->setEvaluator(Plato::Elliptic::evaluator_t::GRAD_X, tGradientX, tName);
        }
        return tMassFunction;
    }

    /******************************************************************************//**
     * \brief Create the 'first mass moment divided by the mass' function (CG)
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aMomentType mass moment type (FirstX, FirstY, FirstZ)
     * \return scalar function base
    **********************************************************************************/
    template<typename PhysicsType>
    std::shared_ptr<Plato::Elliptic::CriterionEvaluatorBase>
    CriterionEvaluatorMassProperties<PhysicsType>::getFirstMomentOverMassRatio(
        const Plato::SpatialModel & aSpatialModel,
        const std::string         & aMomentType
    )
    {
        const std::string tNumeratorName = std::string("CG Numerator (Moment type = ")
                                         + aMomentType + ")";
        std::shared_ptr<Plato::Elliptic::CriterionEvaluatorScalarFunction<PhysicsType>> tNumerator =
             std::make_shared<Plato::Elliptic::CriterionEvaluatorScalarFunction<PhysicsType>>(aSpatialModel, mDataMap);
        tNumerator->setFunctionName(tNumeratorName);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();

            std::shared_ptr<Plato::Elliptic::CriterionMassMoment<Residual>> tNumeratorValue = 
                 std::make_shared<Plato::Elliptic::CriterionMassMoment<Residual>>(tDomain, mDataMap);
            tNumeratorValue->setMaterialDensity(mMaterialDensities[tName]);
            tNumeratorValue->setCalculationType(aMomentType);
            tNumerator->setEvaluator(Plato::Elliptic::evaluator_t::VALUE, tNumeratorValue, tName);

            std::shared_ptr<Plato::Elliptic::CriterionMassMoment<GradientU>> tNumeratorGradientU = 
                 std::make_shared<Plato::Elliptic::CriterionMassMoment<GradientU>>(tDomain, mDataMap);
            tNumeratorGradientU->setMaterialDensity(mMaterialDensities[tName]);
            tNumeratorGradientU->setCalculationType(aMomentType);
            tNumerator->setEvaluator(Plato::Elliptic::evaluator_t::GRAD_U, tNumeratorGradientU, tName);

            std::shared_ptr<Plato::Elliptic::CriterionMassMoment<GradientZ>> tNumeratorGradientZ = 
                 std::make_shared<Plato::Elliptic::CriterionMassMoment<GradientZ>>(tDomain, mDataMap);
            tNumeratorGradientZ->setMaterialDensity(mMaterialDensities[tName]);
            tNumeratorGradientZ->setCalculationType(aMomentType);
            tNumerator->setEvaluator(Plato::Elliptic::evaluator_t::GRAD_Z, tNumeratorGradientZ, tName);

            std::shared_ptr<Plato::Elliptic::CriterionMassMoment<GradientX>> tNumeratorGradientX = 
                 std::make_shared<Plato::Elliptic::CriterionMassMoment<GradientX>>(tDomain, mDataMap);
            tNumeratorGradientX->setMaterialDensity(mMaterialDensities[tName]);
            tNumeratorGradientX->setCalculationType(aMomentType);
            tNumerator->setEvaluator(Plato::Elliptic::evaluator_t::GRAD_X, tNumeratorGradientX, tName);
        }

        const std::string tDenominatorName = std::string("CG Mass Denominator (Moment type = ")
                                           + aMomentType + ")";
        std::shared_ptr<Plato::Elliptic::CriterionEvaluatorScalarFunction<PhysicsType>> tDenominator = 
             getMassFunction(aSpatialModel);
        tDenominator->setFunctionName(tDenominatorName);

        std::shared_ptr<Plato::Elliptic::CriterionEvaluatorDivision<PhysicsType>> tMomentOverMassRatioFunction =
          std::make_shared<Plato::Elliptic::CriterionEvaluatorDivision<PhysicsType>>(aSpatialModel, mDataMap);
        tMomentOverMassRatioFunction->allocateNumeratorFunction(tNumerator);
        tMomentOverMassRatioFunction->allocateDenominatorFunction(tDenominator);
        tMomentOverMassRatioFunction->setFunctionName(std::string("CG ") + aMomentType);
        return tMomentOverMassRatioFunction;
    }

    /******************************************************************************//**
     * \brief Create the second mass moment function
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aMomentType second mass moment type (XX, XY, YY, ...)
     * \return scalar function base
    **********************************************************************************/
    template<typename PhysicsType>
    std::shared_ptr<Plato::Elliptic::CriterionEvaluatorBase>
    CriterionEvaluatorMassProperties<PhysicsType>::getSecondMassMoment(
        const Plato::SpatialModel & aSpatialModel,
        const std::string & aMomentType
    )
    {
        const std::string tInertiaName = std::string("Second Mass Moment (Moment type = ")
                                         + aMomentType + ")";
        std::shared_ptr<Plato::Elliptic::CriterionEvaluatorScalarFunction<PhysicsType>> tSecondMomentFunction =
             std::make_shared<Plato::Elliptic::CriterionEvaluatorScalarFunction<PhysicsType>>(aSpatialModel, mDataMap);
        tSecondMomentFunction->setFunctionName(tInertiaName);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();

            std::shared_ptr<Plato::Elliptic::CriterionMassMoment<Residual>> tValue = 
                 std::make_shared<Plato::Elliptic::CriterionMassMoment<Residual>>(tDomain, mDataMap);
            tValue->setMaterialDensity(mMaterialDensities[tName]);
            tValue->setCalculationType(aMomentType);
            tSecondMomentFunction->setEvaluator(Plato::Elliptic::evaluator_t::VALUE, tValue, tName);

            std::shared_ptr<Plato::Elliptic::CriterionMassMoment<GradientU>> tGradientU = 
                 std::make_shared<Plato::Elliptic::CriterionMassMoment<GradientU>>(tDomain, mDataMap);
            tGradientU->setMaterialDensity(mMaterialDensities[tName]);
            tGradientU->setCalculationType(aMomentType);
            tSecondMomentFunction->setEvaluator(Plato::Elliptic::evaluator_t::GRAD_U, tGradientU, tName);

            std::shared_ptr<Plato::Elliptic::CriterionMassMoment<GradientZ>> tGradientZ = 
                 std::make_shared<Plato::Elliptic::CriterionMassMoment<GradientZ>>(tDomain, mDataMap);
            tGradientZ->setMaterialDensity(mMaterialDensities[tName]);
            tGradientZ->setCalculationType(aMomentType);
            tSecondMomentFunction->setEvaluator(Plato::Elliptic::evaluator_t::GRAD_Z, tGradientZ, tName);

            std::shared_ptr<Plato::Elliptic::CriterionMassMoment<GradientX>> tGradientX = 
                 std::make_shared<Plato::Elliptic::CriterionMassMoment<GradientX>>(tDomain, mDataMap);
            tGradientX->setMaterialDensity(mMaterialDensities[tName]);
            tGradientX->setCalculationType(aMomentType);
            tSecondMomentFunction->setEvaluator(Plato::Elliptic::evaluator_t::GRAD_X, tGradientX, tName);
        }
        return tSecondMomentFunction;
    }

    /******************************************************************************//**
     * \brief Create the moment of inertia function
     * \param [in] aSpatialModel Plato Analyze spatial domain
     * \param [in] aAxes axes about which to compute the moment of inertia (XX, YY, ..)
     * \return scalar function base
    **********************************************************************************/
    template<typename PhysicsType>
    std::shared_ptr<Plato::Elliptic::CriterionEvaluatorBase>
    CriterionEvaluatorMassProperties<PhysicsType>::getMomentOfInertia(
        const Plato::SpatialModel & aSpatialModel,
        const std::string & aAxes
    )
    {
        std::shared_ptr<Plato::Elliptic::CriterionEvaluatorWeightedSum<PhysicsType>> tMomentOfInertiaFunction = 
               std::make_shared<Plato::Elliptic::CriterionEvaluatorWeightedSum<PhysicsType>>(aSpatialModel, mDataMap);
        tMomentOfInertiaFunction->setFunctionName(std::string("Inertia ") + aAxes);

        if (aAxes == "XX")
        {
            tMomentOfInertiaFunction->allocateScalarFunctionBase(
                getSecondMassMoment(aSpatialModel, "SecondYY"));
            tMomentOfInertiaFunction->allocateScalarFunctionBase(
                getSecondMassMoment(aSpatialModel, "SecondZZ"));
            tMomentOfInertiaFunction->appendFunctionWeight(1.0);
            tMomentOfInertiaFunction->appendFunctionWeight(1.0);
        }
        else if (aAxes == "YY")
        {
            tMomentOfInertiaFunction->allocateScalarFunctionBase(
                getSecondMassMoment(aSpatialModel, "SecondXX"));
            tMomentOfInertiaFunction->allocateScalarFunctionBase(
                getSecondMassMoment(aSpatialModel, "SecondZZ"));
            tMomentOfInertiaFunction->appendFunctionWeight(1.0);
            tMomentOfInertiaFunction->appendFunctionWeight(1.0);
        }
        else if (aAxes == "ZZ")
        {
            tMomentOfInertiaFunction->allocateScalarFunctionBase(
                getSecondMassMoment(aSpatialModel, "SecondXX"));
            tMomentOfInertiaFunction->allocateScalarFunctionBase(
                getSecondMassMoment(aSpatialModel, "SecondYY"));
            tMomentOfInertiaFunction->appendFunctionWeight(1.0);
            tMomentOfInertiaFunction->appendFunctionWeight(1.0);
        }
        else if (aAxes == "XY")
        {
            tMomentOfInertiaFunction->allocateScalarFunctionBase(
                getSecondMassMoment(aSpatialModel, "SecondXY"));
            tMomentOfInertiaFunction->appendFunctionWeight(-1.0);
        }
        else if (aAxes == "XZ")
        {
            tMomentOfInertiaFunction->allocateScalarFunctionBase(
                getSecondMassMoment(aSpatialModel, "SecondXZ"));
            tMomentOfInertiaFunction->appendFunctionWeight(-1.0);
        }
        else if (aAxes == "YZ")
        {
            tMomentOfInertiaFunction->allocateScalarFunctionBase(
                getSecondMassMoment(aSpatialModel, "SecondYZ"));
            tMomentOfInertiaFunction->appendFunctionWeight(-1.0);
        }
        else
        {
            const std::string tErrorString = std::string("Specified axes '") +
            aAxes + "' not implemented for moment of inertia calculation. " 
                          + "Options are: XX, YY, ZZ, XY, XZ, YZ";
            ANALYZE_THROWERR(tErrorString)
        }

        return tMomentOfInertiaFunction;
    }

    /******************************************************************************//**
     * \brief Create the moment of inertia function about the CG in the principal coordinate frame
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aAxes axes about which to compute the moment of inertia (XX, YY, ..)
     * \return scalar function base
    **********************************************************************************/
    template<typename PhysicsType>
    std::shared_ptr<Plato::Elliptic::CriterionEvaluatorBase>
    CriterionEvaluatorMassProperties<PhysicsType>::getMomentOfInertiaRotatedAboutCG(
        const Plato::SpatialModel & aSpatialModel,
        const std::string         & aAxes
    )
    {
        std::shared_ptr<Plato::Elliptic::CriterionEvaluatorWeightedSum<PhysicsType>> tMomentOfInertiaFunction = 
               std::make_shared<Plato::Elliptic::CriterionEvaluatorWeightedSum<PhysicsType>>(aSpatialModel, mDataMap);
        tMomentOfInertiaFunction->setFunctionName(std::string("InertiaRot ") + aAxes);

        std::vector<Plato::Scalar> tInertiaWeights(6);
        Plato::Scalar tMassWeight;

        getInertiaAndMassWeights(tInertiaWeights, tMassWeight, aAxes);
        for (unsigned int tIndex = 0; tIndex < 6; ++tIndex)
            tMomentOfInertiaFunction->appendFunctionWeight(tInertiaWeights[tIndex]);

        tMomentOfInertiaFunction->allocateScalarFunctionBase(getMomentOfInertia(aSpatialModel, "XX"));
        tMomentOfInertiaFunction->allocateScalarFunctionBase(getMomentOfInertia(aSpatialModel, "YY"));
        tMomentOfInertiaFunction->allocateScalarFunctionBase(getMomentOfInertia(aSpatialModel, "ZZ"));
        tMomentOfInertiaFunction->allocateScalarFunctionBase(getMomentOfInertia(aSpatialModel, "XY"));
        tMomentOfInertiaFunction->allocateScalarFunctionBase(getMomentOfInertia(aSpatialModel, "XZ"));
        tMomentOfInertiaFunction->allocateScalarFunctionBase(getMomentOfInertia(aSpatialModel, "YZ"));

        tMomentOfInertiaFunction->allocateScalarFunctionBase(getMassFunction(aSpatialModel));
        tMomentOfInertiaFunction->appendFunctionWeight(tMassWeight);

        return tMomentOfInertiaFunction;
    }

    /******************************************************************************//**
     * \brief Compute the inertia weights and mass weight for the inertia about the CG rotated into principal frame
     * \param [out] aInertiaWeights inertia weights
     * \param [out] aMassWeight mass weight
     * \param [in] aAxes axes about which to compute the moment of inertia (XX, YY, ..)
    **********************************************************************************/
    template<typename PhysicsType>
    void
    CriterionEvaluatorMassProperties<PhysicsType>::getInertiaAndMassWeights(std::vector<Plato::Scalar> & aInertiaWeights, 
                             Plato::Scalar & aMassWeight, 
                             const std::string & aAxes)
    {
        const Plato::Scalar Q11 = mInertiaRotationMatrix(0,0);
        const Plato::Scalar Q12 = mInertiaRotationMatrix(0,1);
        const Plato::Scalar Q13 = mInertiaRotationMatrix(0,2);

        const Plato::Scalar Q21 = mInertiaRotationMatrix(1,0);
        const Plato::Scalar Q22 = mInertiaRotationMatrix(1,1);
        const Plato::Scalar Q23 = mInertiaRotationMatrix(1,2);

        const Plato::Scalar Q31 = mInertiaRotationMatrix(2,0);
        const Plato::Scalar Q32 = mInertiaRotationMatrix(2,1);
        const Plato::Scalar Q33 = mInertiaRotationMatrix(2,2);

        if (aAxes == "XX")
        {
            aInertiaWeights[0] = Q11*Q11;
            aInertiaWeights[1] = Q21*Q21;
            aInertiaWeights[2] = Q31*Q31;
            aInertiaWeights[3] = 2.0*Q11*Q21;
            aInertiaWeights[4] = 2.0*Q11*Q31;
            aInertiaWeights[5] = 2.0*Q21*Q31;

            aMassWeight = mMinusRotatedParallelAxisTheoremMatrix(0,0);
        }
        else if (aAxes == "YY")
        {
            aInertiaWeights[0] =  Q12*Q12;
            aInertiaWeights[1] =  Q22*Q22;
            aInertiaWeights[2] =  Q32*Q32;
            aInertiaWeights[3] =  2.0*Q12*Q22;
            aInertiaWeights[4] =  2.0*Q12*Q32;
            aInertiaWeights[5] =  2.0*Q22*Q32;

            aMassWeight = mMinusRotatedParallelAxisTheoremMatrix(1,1);
        }
        else if (aAxes == "ZZ")
        {
            aInertiaWeights[0] =  Q13*Q13;
            aInertiaWeights[1] =  Q23*Q23;
            aInertiaWeights[2] =  Q33*Q33;
            aInertiaWeights[3] =  2.0*Q13*Q23;
            aInertiaWeights[4] =  2.0*Q13*Q33;
            aInertiaWeights[5] =  2.0*Q23*Q33;

            aMassWeight = mMinusRotatedParallelAxisTheoremMatrix(2,2);
        }
        else if (aAxes == "XY")
        {
            aInertiaWeights[0] =  Q11*Q12;
            aInertiaWeights[1] =  Q21*Q22;
            aInertiaWeights[2] =  Q31*Q32;
            aInertiaWeights[3] =  Q11*Q22 + Q12*Q21;
            aInertiaWeights[4] =  Q11*Q32 + Q12*Q31;
            aInertiaWeights[5] =  Q21*Q32 + Q22*Q31;

            aMassWeight = mMinusRotatedParallelAxisTheoremMatrix(0,1);
        }
        else if (aAxes == "XZ")
        {
            aInertiaWeights[0] =  Q11*Q13;
            aInertiaWeights[1] =  Q21*Q23;
            aInertiaWeights[2] =  Q31*Q33;
            aInertiaWeights[3] =  Q11*Q23 + Q13*Q21;
            aInertiaWeights[4] =  Q11*Q33 + Q13*Q31;
            aInertiaWeights[5] =  Q21*Q33 + Q23*Q31;

            aMassWeight = mMinusRotatedParallelAxisTheoremMatrix(0,2);
        }
        else if (aAxes == "YZ")
        {
            aInertiaWeights[0] =  Q12*Q13;
            aInertiaWeights[1] =  Q22*Q23;
            aInertiaWeights[2] =  Q32*Q33;
            aInertiaWeights[3] =  Q12*Q23 + Q13*Q22;
            aInertiaWeights[4] =  Q12*Q33 + Q13*Q32;
            aInertiaWeights[5] =  Q22*Q33 + Q23*Q32;

            aMassWeight = mMinusRotatedParallelAxisTheoremMatrix(1,2);
        }
        else
        {
            const std::string tErrorString = std::string("Specified axes '") +
            aAxes + "' not implemented for inertia and mass weights calculation. " 
                          + "Options are: XX, YY, ZZ, XY, XZ, YZ";
            ANALYZE_THROWERR(tErrorString)
        }
    }

    /******************************************************************************//**
     * \brief Primary Mass Properties Function constructor
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap PLATO Engine and Analyze data map
     * \param [in] aInputParams input parameters database
     * \param [in] aName user defined function name
    **********************************************************************************/
    template<typename PhysicsType>
    CriterionEvaluatorMassProperties<PhysicsType>::CriterionEvaluatorMassProperties(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
              std::string            & aName
    ) :
      mSpatialModel (aSpatialModel),
      mDataMap      (aDataMap),
      mFunctionName (aName)
    {
      initialize(aInputParams);
    }

    /******************************************************************************//**
     * \brief Compute the X, Y, and Z extents of the mesh (e.g. (X_max - X_min))
     * \param [in] aMesh mesh database
    **********************************************************************************/
    template<typename PhysicsType>
    void
    CriterionEvaluatorMassProperties<PhysicsType>::computeMeshExtent(Plato::Mesh aMesh)
    {
        auto tNodeCoordinates = aMesh->Coordinates();
        auto tSpaceDim        = aMesh->NumDimensions();
        auto tNumVertices     = aMesh->NumNodes();

        assert(tSpaceDim == 3);

        Plato::ScalarVector tXCoordinates("X-Coordinates", tNumVertices);
        Plato::ScalarVector tYCoordinates("Y-Coordinates", tNumVertices);
        Plato::ScalarVector tZCoordinates("Z-Coordinates", tNumVertices);

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumVertices), KOKKOS_LAMBDA(const Plato::OrdinalType & tVertexIndex)
        {
            const Plato::Scalar x_coordinate = tNodeCoordinates[tVertexIndex * tSpaceDim + 0];
            const Plato::Scalar y_coordinate = tNodeCoordinates[tVertexIndex * tSpaceDim + 1];
            const Plato::Scalar z_coordinate = tNodeCoordinates[tVertexIndex * tSpaceDim + 2];

            tXCoordinates(tVertexIndex) = x_coordinate;
            tYCoordinates(tVertexIndex) = y_coordinate;
            tZCoordinates(tVertexIndex) = z_coordinate;
        }, "Fill vertex coordinate views");

        Plato::Scalar tXmin;
        Plato::Scalar tXmax;
        Plato::blas1::min(tXCoordinates, tXmin);
        Plato::blas1::max(tXCoordinates, tXmax);

        Plato::Scalar tYmin;
        Plato::Scalar tYmax;
        Plato::blas1::min(tYCoordinates, tYmin);
        Plato::blas1::max(tYCoordinates, tYmax);

        Plato::Scalar tZmin;
        Plato::Scalar tZmax;
        Plato::blas1::min(tZCoordinates, tZmin);
        Plato::blas1::max(tZCoordinates, tZmax);

        mMeshExtentX = std::abs(tXmax - tXmin);
        mMeshExtentY = std::abs(tYmax - tYmin);
        mMeshExtentZ = std::abs(tZmax - tZmin);
    }

    template<typename PhysicsType>
    bool 
    CriterionEvaluatorMassProperties<PhysicsType>::
    isLinear() 
    const
    {
      return ( mLeastSquaresFunction->isLinear() );
    }

    template<typename PhysicsType>
    void
    CriterionEvaluatorMassProperties<PhysicsType>::
    updateProblem(
      const Plato::Database & aDatabase,
      const Plato::Scalar   & aCycle
    ) const
    {
      mLeastSquaresFunction->updateProblem(aDatabase,aCycle);
    }

    template<typename PhysicsType>
    Plato::Scalar
    CriterionEvaluatorMassProperties<PhysicsType>::
    value(const Plato::Database & aDatabase,
          const Plato::Scalar   & aCycle
    ) const
    {
      Plato::Scalar tFunctionValue = mLeastSquaresFunction->value(aDatabase,aCycle);
      return tFunctionValue;
    }

    template<typename PhysicsType>
    Plato::ScalarVector
    CriterionEvaluatorMassProperties<PhysicsType>::
    gradientState(
      const Plato::Database & aDatabase,
      const Plato::Scalar   & aCycle
    ) const
    {
      Plato::ScalarVector tGradientU = mLeastSquaresFunction->gradientState(aDatabase,aCycle);
      return tGradientU;
    }

    template<typename PhysicsType>
    Plato::ScalarVector
    CriterionEvaluatorMassProperties<PhysicsType>::
    gradientConfig(
      const Plato::Database & aDatabase,
      const Plato::Scalar   & aCycle
    ) const
    {
      Plato::ScalarVector tGradientX = mLeastSquaresFunction->gradientConfig(aDatabase,aCycle);
      return tGradientX;
    }

    template<typename PhysicsType>
    Plato::ScalarVector
    CriterionEvaluatorMassProperties<PhysicsType>::
    gradientControl(
      const Plato::Database & aDatabase,
      const Plato::Scalar   & aCycle
    ) const
    {
      Plato::ScalarVector tGradientZ = mLeastSquaresFunction->gradientControl(aDatabase,aCycle);
      return tGradientZ;
    }

    template<typename PhysicsType>
    std::string CriterionEvaluatorMassProperties<PhysicsType>::
    name() const
    {
      return mFunctionName;
    }
} // namespace Elliptic

} // namespace Plato
