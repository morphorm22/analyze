#include <fstream>
#include <string>
#include <sstream>

#include "ParseTools.hpp"
#include "SpatialModel.hpp"
#include "PlatoUtilities.hpp"
#include "AnalyzeAppUtils.hpp"

namespace Plato
{
  inline
  std::string
  getFileExtension(std::string const & aFileName)
  {
    if( std::count(aFileName.begin(), aFileName.end(), '.') == 0 )
    {
      std::stringstream ss;
      ss << "Filename (" << aFileName << ") does not have an extension";
      ANALYZE_THROWERR(ss.str());
    }

    auto tPos = aFileName.rfind('.');
    return aFileName.substr(tPos+1, std::string::npos);
  }

  inline
  std::string
  trim(std::string const & aLine)
  {
    const char* tWhiteSpace = " \t\v\r\n";
    auto tStart = aLine.find_first_not_of(tWhiteSpace);
    if(tStart == std::string::npos) return std::string();
    auto tEnd = aLine.find_last_not_of(tWhiteSpace);
    return aLine.substr(tStart, tEnd - tStart + 1);
  }

  class CSVReader
  {
    Teuchos::Array<std::string> mColumnNames;
    std::vector<Plato::HostScalarVector> mHostData;
    std::vector<Plato::ScalarVector> mDeviceData;
    Plato::DataMap & mDataMap;

    public:
    CSVReader(
      Teuchos::ParameterList & aInputs,
      Plato::DataMap         & aDataMap,
      Plato::Mesh              aMesh
    ) : mDataMap(aDataMap)
    {
      mColumnNames = Plato::ParseTools::getParam<Teuchos::Array<std::string>>(aInputs, "Columns");

      auto tType = Plato::ParseTools::getParam<std::string>(aInputs, "Centering", "Element");
      Plato::OrdinalType tNumData;
      if( tType == "Element" )
      {
        tNumData = aMesh->NumElements();
      }
      else
      if( tType == "Node" )
      {
        tNumData = aMesh->NumNodes();
      }

      for( auto & tColumnName : mColumnNames )
      {
        if( aDataMap.scalarVectors.count(tColumnName) )
        {
          std::stringstream ss;
          ss << "Attempted to overwrite '" << tColumnName << "' in DataMap";
          ANALYZE_THROWERR(ss.str());
        }

        Plato::ScalarVector tDeviceData(tColumnName, tNumData);
        mDeviceData.push_back(tDeviceData);
        aDataMap.scalarVectors[tColumnName] = tDeviceData;

        Plato::HostScalarVector tHostData(tColumnName, tNumData);
        mHostData.push_back(tHostData);
      }
    }

    inline
    void
    close()
    {
      for(int i=0; i<mColumnNames.size(); i++)
      {
        Kokkos::deep_copy(mDeviceData[i], mHostData[i]);
      }
    }

    inline
    void
    readLine(std::string const & aLine)
    {
      // ignore any empty lines
      if( aLine.size() == 0 ) return;

      // ignore any lines where the first non-space character is '#'
      auto tTrimmed = Plato::trim(aLine);
      if( tTrimmed.size() && tTrimmed[0] == '#' ) return;

      // split line by ','
      std::vector<std::string> tRawTokens = Plato::split(aLine, ',');

      // trim tokens (remove whitespace at each end)
      std::vector<std::string> tTokens;
      for(auto const & tToken : tRawTokens)
      {
        auto tTrimmed = Plato::trim(tToken);
        if(tTrimmed.size() == 0) ANALYZE_THROWERR("Encountered token of zero length.");
        tTokens.push_back(tTrimmed);
      }

      // convert index (first column)
      int tIndex = stoi(tTokens[0]);

      // convert values
      for(int i=1; i<tTokens.size(); i++)
      {
        mHostData[i](tIndex) = stod(tTokens[i]);
      }
    }
  };

  void 
  readCSVData(
    std::string            const & aFileName,
    Teuchos::ParameterList       & aInputs,
    Plato::DataMap               & aDataMap,
    Plato::Mesh                    aMesh
  );

  Plato::ScalarArray3D
  createOrthonormalTensor3D(
    Teuchos::ParameterList & tTransformParams,
    Plato::DataMap         & aDataMap
  );

  Plato::ScalarArray3D
  createOrthonormalTensor2D(
    Teuchos::ParameterList & tTransformParams,
    Plato::DataMap         & aDataMap
  );

  Plato::ScalarArray3D
  createOrthonormalTensor(
    Teuchos::ParameterList & tTransformParams,
    Plato::DataMap         & aDataMap,
    Plato::OrdinalType       aDimension
  );

  void
  readInputData(
    Teuchos::ParameterList & aInputs,
    Plato::DataMap         & aDataMap,
    Plato::Mesh              aMesh
  );


  void
  verifyOrthogonality(
    Plato::ScalarVector a1X,
    Plato::ScalarVector a1Y,
    Plato::ScalarVector a1Z,
    Plato::ScalarVector a2X,
    Plato::ScalarVector a2Y,
    Plato::ScalarVector a2Z,
    Plato::Scalar       aTol
  );

  void
  verifyOrthogonality(
    Plato::ScalarVector a1X,
    Plato::ScalarVector a1Y,
    Plato::ScalarVector a2X,
    Plato::ScalarVector a2Y,
    Plato::Scalar       aTol
  );

  void
  normalizeVector(
    Plato::ScalarVector aX,
    Plato::ScalarVector aY,
    Plato::ScalarVector aZ
  );

  void
  normalizeVector(
    Plato::ScalarVector aX,
    Plato::ScalarVector aY
  );

}
