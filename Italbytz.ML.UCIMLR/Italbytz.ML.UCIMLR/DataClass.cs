using Microsoft.ML;

namespace Italbytz.ML.UCIMLR;

public class DataClass(DatasetEnum datasetEnum)
{
    private IDataView dataView;

    public IDataView DataView
    {
        get
        {
            if (dataView == null) dataView = GetDataView();
            return dataView;
        }
    }

    private IDataView? GetDataView()
    {
        // ToDo: Implement the logic to load the data from the dataset
        throw new NotImplementedException();
    }

    public static DataClass Get(DatasetEnum datasetEnum)
    {
        return new DataClass(datasetEnum);
    }
}