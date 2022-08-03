/*
 * CubicMicromorphicLinearElasticMaterial.cpp
 *
 *  Created on: Oct 18, 2021
 */

#include "hyperbolic/micromorphic/CubicMicromorphicLinearElasticMaterial.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Linear elastic Cubic micromorphic material model constructor. - 1D
**********************************************************************************/
template<>
Plato::CubicMicromorphicLinearElasticMaterial<1>::
CubicMicromorphicLinearElasticMaterial(const Teuchos::ParameterList& paramList) :
    Plato::MicromorphicLinearElasticMaterial<1>(paramList)
{
    Plato::Scalar tLambda_e  = paramList.get<Plato::Scalar>("Lambda_e");
    Plato::Scalar tMu_e      = paramList.get<Plato::Scalar>("Mu_e");
    mCellStiffnessCe(0, 0)   = tLambda_e + 2.0 * tMu_e;

    Plato::Scalar tMu_c      = paramList.get<Plato::Scalar>("Mu_c");
    mCellStiffnessCc(0, 0)   = tMu_c;

    Plato::Scalar tLambda_m  = paramList.get<Plato::Scalar>("Lambda_m");
    Plato::Scalar tMu_m      = paramList.get<Plato::Scalar>("Mu_m");
    mCellStiffnessCm(0, 0)   = tLambda_m + 2.0 * tMu_m;
}

/******************************************************************************//**
 * \brief Linear elastic Cubic micromorphic material model constructor. - 2D
**********************************************************************************/
template<>
Plato::CubicMicromorphicLinearElasticMaterial<2>::
CubicMicromorphicLinearElasticMaterial(const Teuchos::ParameterList& paramList) :
    Plato::MicromorphicLinearElasticMaterial<2>(paramList)
{
    Plato::Scalar tLambda_e  = paramList.get<Plato::Scalar>("Lambda_e");
    Plato::Scalar tMu_e      = paramList.get<Plato::Scalar>("Mu_e");
    Plato::Scalar tMu_star_e = paramList.get<Plato::Scalar>("Mu_star_e");
    mCellStiffnessCe(0, 0)   = tLambda_e + 2.0 * tMu_e;
    mCellStiffnessCe(0, 1)   = tLambda_e;
    mCellStiffnessCe(1, 0)   = tLambda_e;
    mCellStiffnessCe(1, 1)   = tLambda_e + 2.0 * tMu_e;
    mCellStiffnessCe(2, 2)   = tMu_star_e;

    Plato::Scalar tMu_c      = paramList.get<Plato::Scalar>("Mu_c");
    mCellStiffnessCc(0, 0)   = tMu_c;

    Plato::Scalar tLambda_m  = paramList.get<Plato::Scalar>("Lambda_m");
    Plato::Scalar tMu_m      = paramList.get<Plato::Scalar>("Mu_m");
    Plato::Scalar tMu_star_m = paramList.get<Plato::Scalar>("Mu_star_m");
    mCellStiffnessCm(0, 0)   = tLambda_m + 2.0 * tMu_m;
    mCellStiffnessCm(0, 1)   = tLambda_m;
    mCellStiffnessCm(1, 0)   = tLambda_m;
    mCellStiffnessCm(1, 1)   = tLambda_m + 2.0 * tMu_m;
    mCellStiffnessCm(2, 2)   = tMu_star_m;
}

/******************************************************************************//**
 * \brief Linear elastic Cubic micromorphic material model constructor. - 3D
**********************************************************************************/
template<>
Plato::CubicMicromorphicLinearElasticMaterial<3>::
CubicMicromorphicLinearElasticMaterial(const Teuchos::ParameterList& paramList) :
    Plato::MicromorphicLinearElasticMaterial<3>(paramList)
{
    Plato::Scalar tLambda_e  = paramList.get<Plato::Scalar>("Lambda_e");
    Plato::Scalar tMu_e      = paramList.get<Plato::Scalar>("Mu_e");
    Plato::Scalar tMu_star_e = paramList.get<Plato::Scalar>("Mu_star_e");
    mCellStiffnessCe(0, 0)   = tLambda_e + 2.0 * tMu_e;
    mCellStiffnessCe(0, 1)   = tLambda_e;
    mCellStiffnessCe(0, 2)   = tLambda_e;
    mCellStiffnessCe(1, 0)   = tLambda_e;
    mCellStiffnessCe(1, 1)   = tLambda_e + 2.0 * tMu_e;
    mCellStiffnessCe(1, 2)   = tLambda_e;
    mCellStiffnessCe(2, 0)   = tLambda_e;
    mCellStiffnessCe(2, 1)   = tLambda_e;
    mCellStiffnessCe(2, 2)   = tLambda_e + 2.0 * tMu_e;
    mCellStiffnessCe(3, 3)   = tMu_star_e;
    mCellStiffnessCe(4, 4)   = tMu_star_e;
    mCellStiffnessCe(5, 5)   = tMu_star_e;

    Plato::Scalar tMu_c      = paramList.get<Plato::Scalar>("Mu_c");
    mCellStiffnessCc(0, 0)   = tMu_c;
    mCellStiffnessCc(1, 1)   = tMu_c;
    mCellStiffnessCc(2, 2)   = tMu_c;

    Plato::Scalar tLambda_m  = paramList.get<Plato::Scalar>("Lambda_m");
    Plato::Scalar tMu_m      = paramList.get<Plato::Scalar>("Mu_m");
    Plato::Scalar tMu_star_m = paramList.get<Plato::Scalar>("Mu_star_m");
    mCellStiffnessCm(0, 0)   = tLambda_m + 2.0 * tMu_m;
    mCellStiffnessCm(0, 1)   = tLambda_m;
    mCellStiffnessCm(0, 2)   = tLambda_m;
    mCellStiffnessCm(1, 0)   = tLambda_m;
    mCellStiffnessCm(1, 1)   = tLambda_m + 2.0 * tMu_m;
    mCellStiffnessCm(1, 2)   = tLambda_m;
    mCellStiffnessCm(2, 0)   = tLambda_m;
    mCellStiffnessCm(2, 1)   = tLambda_m;
    mCellStiffnessCm(2, 2)   = tLambda_m + 2.0 * tMu_m;
    mCellStiffnessCm(3, 3)   = tMu_star_m;
    mCellStiffnessCm(4, 4)   = tMu_star_m;
    mCellStiffnessCm(5, 5)   = tMu_star_m;
}

}
// namespace Plato
