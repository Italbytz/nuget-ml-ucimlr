using Italbytz.ML.Data;
using Microsoft.ML;

namespace Italbytz.ML.UCIMLR;

public interface IDataset
{
    public IDataView DataView { get; }

    public IEnumerable<TrainValidateTestFileNames> GetTrainValidateTestFiles(
        string saveFolderPath,
        string filePrefix,
        string? samplingKeyColumnName = null,
        double validateFraction = 0.15,
        double testFraction = 0.15,
        int[]? seeds = null
    );
}