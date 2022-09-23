#ifndef PLATO_TMKINETICS_FACTORY_HPP
#define PLATO_TMKINETICS_FACTORY_HPP

#include "LinearTMKinetics.hpp"
#include "NonLinearTMKinetics.hpp"
#include "ExpressionTMKinetics.hpp"

namespace Plato
{
/******************************************************************************//**
 * \brief TMKinetics Factory for creating TMKinetics variants.
**********************************************************************************/
template<typename EvaluationType, typename ElementType>
class TMKineticsFactory
{
public:
    /******************************************************************************//**
    * \brief TMKinetics factory constructor.
    **********************************************************************************/
    TMKineticsFactory() {}

    /******************************************************************************//**
    * \brief Create a TMKinetics functor.
    * \param [in] aMaterialInfo - a material element stiffness matrix or
                                  a material model interface
    * \param [in] aParamList - input parameter list
    * \return Teuchos reference counter pointer to the linear stress functor
    **********************************************************************************/
    Teuchos::RCP<Plato::AbstractTMKinetics<EvaluationType, ElementType> > create(
        const Teuchos::RCP<Plato::MaterialModel<ElementType::mNumSpatialDims>>   aMaterialModel,
        const Plato::SpatialDomain                                             & aSpatialDomain,
        const Plato::DataMap                                                   & aDataMap
    )
    {
        Plato::MaterialModelType tModelType = aMaterialModel->type();
        if (tModelType == Plato::MaterialModelType::Nonlinear)
        {
            return Teuchos::rcp( new Plato::NonLinearTMKinetics<EvaluationType, ElementType> (aMaterialModel, aSpatialDomain, aDataMap) );
        } 
        else if (tModelType == Plato::MaterialModelType::Linear)
        {
            return Teuchos::rcp( new Plato::LinearTMKinetics<EvaluationType, ElementType>(aMaterialModel, aSpatialDomain, aDataMap) );
        } 
        else if (tModelType == Plato::MaterialModelType::Expression)
        {
            return Teuchos::rcp( new Plato::ExpressionTMKinetics<EvaluationType, ElementType> (aMaterialModel, aSpatialDomain, aDataMap) );
        }
        else
        {
            ANALYZE_THROWERR("Unknown Material Model Type")
        }
    }
};
// class TMKineticsFactory

}// namespace Plato
#endif
