#pragma once

#include <memory>
#include <map>

#include "EngineMesh.hpp"
#include "AbstractPlatoMeshIO.hpp"

#include "mesh/ExodusIO.hpp"

namespace Plato
{
    using PairList = std::vector<std::pair<std::string, Plato::OrdinalType>>;
    using StrList = std::vector<std::string>;
    using OrdList = std::vector<Plato::OrdinalType>;

    class EngineMeshIO : public AbstractMeshIO
    {
        Plato::EngineMesh & mMesh;
        std::shared_ptr<ExodusIO> mMeshIO;

        PairList mScalarSuffixes;
        PairList mVectorSuffixes;
        PairList mSymtensorSuffixes;

        std::map<std::string, std::vector<std::string>> mVectorNodeVars;

        void setVariableTypeSuffixes();
        void parseNodeVarNames();

        void readNodeScalar( const std::string & aVariableName, Plato::OrdinalType aStepIndex, Plato::ScalarVector aNodeData);

        bool mVariablesAreSet;
        Plato::OrdinalType mPlotIndex;

        struct Variable {
            Variable() {}
            Variable( Plato::ScalarMultiVector Data, StrList DofNames ) :
              isStale(false), Data(Data), DofNames(DofNames) {}

            bool isStale;
            Plato::ScalarMultiVector Data;
            StrList DofNames;
            OrdList VarIndices;
        };

        void setNodeData(std::string aName, Variable & aVar);
        void addNodeData(std::string aName, Variable & aVar);

        void setElementData(std::string aName, Variable & aVar);
        void addElementData(std::string aName, Variable & aVar);

        std::map<std::string, Variable> mNodeVars;
        std::map<std::string, Variable> mElementVars;

        public:
            EngineMeshIO(std::string aOutputFilePath, Plato::EngineMesh & aMesh, std::string aMode="Write");
            ~EngineMeshIO();

            Plato::OrdinalType NumNodes() const override;
            Plato::OrdinalType NumElements() const override;

            void AddNodeData(std::string aName, Plato::ScalarVector aData) override;
            void AddNodeData(std::string aName, Plato::ScalarVector aData, StrList aDofNames) override;
            void AddNodeData(std::string aName, Plato::ScalarVector aData, Plato::OrdinalType aNumDofs) override;

            void AddElementData(std::string aName, Plato::ScalarVector aData) override;
            void AddElementData(std::string aName, Plato::ScalarMultiVector aData) override;
            void AddElementData(std::string aName, Plato::ScalarMultiVector aData, StrList aNames);

            void Write(Plato::OrdinalType aStepIndex, Plato::Scalar aTimeValue) override;

            Plato::OrdinalType NumTimeSteps() override;

            Plato::ScalarVector ReadNodeData( const std::string & aVariableName, Plato::OrdinalType aStepIndex) override;

            void
            extractColumn(const Plato::ScalarMultiVector & aFromVector,
                                Plato::OrdinalType         aColIndex,
                          const Plato::ScalarVector      & aToVector);

    };
}
