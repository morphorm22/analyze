#pragma once

#include <string>
#include <vector>

namespace Plato
{

class ExodusIO
{

  public:

    using Int = int;
    using Real = double;

  private:

    struct ElementBlock
    {
        std::string name;
        Int ID;
        Int numNPE;
        Int numElems;
        std::vector<Int> connectivity;
        std::string elementType;
        std::vector<std::vector<Int>> faceGraph;
    };
    std::vector<ElementBlock> mElementBlocks;

    struct NodeSet
    {
        std::string name;
        Int ID;
        std::vector<Int> nodes;
    };
    std::vector<NodeSet> mNodeSets;

    struct SideSet
    {
        std::string name;
        Int ID;
        Int numNPF;
        std::vector<Int> sides;
        std::vector<Int> elems;
        std::vector<Int> nodes;
        
    };
    std::vector<SideSet> mSideSets;

    Int mNumNodes;
    Int mNumElems;
    Int mNumDims;

    Int mFileID;

    std::vector<Int> mNodeGlobalIds;
    std::vector<Int> mElemGlobalIds;

    std::vector<std::vector<Real>> mCoords;

    bool mIgnoreElemMap;
    bool mIgnoreNodeMap;

  public:
    ExodusIO() : mFileID(-1) {}
    ~ExodusIO() { closeMesh(); }
    void readMesh(const std::string & aFileName, bool aIgnoreNodeMap, bool aIgnoreElemMap);

    Int getNumNodes() const { return mNumNodes; }
    Int getNumElems() const { return mNumElems; }

    Int getNumElemBlks() const;
    Int getNumElemInBlk(Int blk) const;
    Int getNnpeInBlk(Int blk) const;
    Int getDimensions() const;

    const std::vector<Int> & getElemToNodeConnInBlk(Int blk) const;

    const std::vector<std::vector<double>> & getCoords() const;

    std::string getElemTypeInBlk(Int blk) const;
    std::string getBlockName(Int blk) const;

    Int getNumNodeSets() const;
    Int getNodeSetLength(Int i) const;
    std::string getNodeSetName(Int i) const;
    const std::vector<Int> & getNodeSetNodes(Int i) const;

    Int getNumSideSets() const;
    Int getSideSetLength(Int i) const;
    Int getSideSetNodesPerFace(Int i) const;
    std::string getSideSetName(Int i) const;
    const std::vector<Int> & getSideSetFaces(Int i) const;
    const std::vector<Int> & getSideSetElems(Int i) const;
    const std::vector<Int> & getSideSetNodes(Int i) const;

    std::vector<std::string> getNodeVarNames() const;
    void writeTime(Int aTimeStep, Real aTimeValue) const;
    void initVars(std::string aCentering, Int aNumVars, std::vector<std::string> aNames ) const;
    void writeNodePlot(Real* aData, Int aVariableIndex, Int aStepIndex) const;
    void writeElemPlot(Real* aData, Int aVariableIndex, Int aStepIndex) const;
    Int getNumSteps() const;
    void readNodePlot(double* aData, std::string aVariableName, Int aStepIndex) const;

    void openMesh(std::string aFileName, const std::string & aMode);

    std::vector<std::vector<Int>> getFaceGraph(std::string) const;
    std::vector<std::vector<Int>> getFaceGraph(Int aBlockIndex) const;

  private:
    void readData();
    void closeMesh();

    void readNodeIds();
    void readElemIds();
    void readNodeSets();
    void readSideSets();
    void readElementBlocks();
    void readCoords();

    void writeData();

    void checkAddExtension(std::string & aName, std::string aExt);

};

} // end namespace Plato
