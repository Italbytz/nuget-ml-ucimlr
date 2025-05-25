using Italbytz.ML.Data;
using Italbytz.ML.ModelBuilder.Configuration;
using Microsoft.ML;

namespace Italbytz.ML.UCIMLR;

public interface IDataset
{
    public IDataView DataView { get; }

    public IColumnProperties[] ColumnProperties { get; }

    public IEnumerable<TrainValidateTestFileNames> GetTrainValidateTestFiles(
        string saveFolderPath,
        string filePrefix,
        string? samplingKeyColumnName = null,
        double validateFraction = 0.15,
        double testFraction = 0.15,
        int[]? seeds = null
    );
}