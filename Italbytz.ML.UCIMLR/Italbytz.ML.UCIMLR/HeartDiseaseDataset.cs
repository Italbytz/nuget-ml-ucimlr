using Italbytz.ML.ModelBuilder.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Italbytz.ML.UCIMLR;

public class
    HeartDiseaseDataset : Dataset<HeartDiseaseDataset.HeartDiseaseModelInput>
{
    public override IEstimator<ITransformer> BuildPipeline(MLContext mlContext,
        ScenarioType scenarioType,
        IEstimator<ITransformer> estimator)
    {
        throw new NotImplementedException();
    }

    public override IDataView LoadFromTextFile(string path,
        char separatorChar = IDataset.TextLoaderDefaults.Separator,
        bool hasHeader = IDataset.TextLoaderDefaults.HasHeader,
        bool allowQuoting = IDataset.TextLoaderDefaults.AllowQuoting,
        bool trimWhitespace = IDataset.TextLoaderDefaults.TrimWhitespace,
        bool allowSparse = IDataset.TextLoaderDefaults.AllowSparse)
    {
        return LoadFromTextFile<HeartDiseaseModelInput>(path, separatorChar,
            hasHeader,
            allowQuoting, trimWhitespace, allowSparse);
    }


    public class HeartDiseaseModelInput
    {
        [LoadColumn(0)] [ColumnName(@"age")] public float Age { get; set; }

        [LoadColumn(1)] [ColumnName(@"sex")] public float Sex { get; set; }

        [LoadColumn(2)] [ColumnName(@"cp")] public float Cp { get; set; }

        [LoadColumn(3)]
        [ColumnName(@"trestbps")]
        public float Trestbps { get; set; }

        [LoadColumn(4)] [ColumnName(@"chol")] public float Chol { get; set; }

        [LoadColumn(5)] [ColumnName(@"fbs")] public float Fbs { get; set; }

        [LoadColumn(6)]
        [ColumnName(@"restecg")]
        public float Restecg { get; set; }

        [LoadColumn(7)]
        [ColumnName(@"thalach")]
        public float Thalach { get; set; }

        [LoadColumn(8)] [ColumnName(@"exang")] public float Exang { get; set; }

        [LoadColumn(9)]
        [ColumnName(@"oldpeak")]
        public float Oldpeak { get; set; }

        [LoadColumn(10)]
        [ColumnName(@"slope")]
        public float Slope { get; set; }

        [LoadColumn(11)] [ColumnName(@"ca")] public float Ca { get; set; }

        [LoadColumn(12)] [ColumnName(@"thal")] public float Thal { get; set; }

        [LoadColumn(13)] [ColumnName(@"num")] public float Num { get; set; }
    }
}