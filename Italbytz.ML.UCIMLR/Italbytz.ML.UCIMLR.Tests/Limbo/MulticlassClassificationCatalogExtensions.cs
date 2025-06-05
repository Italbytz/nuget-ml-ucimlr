using System.Globalization;
using System.Text;
using Italbytz.ML.Data;
using Microsoft.ML;

namespace Italbytz.ML;

public static class MulticlassClassificationCatalogExtension
{
    public static string GetPermutationFeatureImportanceTable(
        this MulticlassClassificationCatalog catalog,
        ITransformer model, IDataView data, string? labelColumnName,
        Metric metric)
    {
        var sb = new StringBuilder();
        sb.AppendLine(
            "Feature, Importance");
        var permutationFeatureImportance =
            catalog
                .PermutationFeatureImportance(
                    model,
                    data,
                    labelColumnName);
        foreach (var (key, value) in permutationFeatureImportance)
        {
            var metricValue = metric switch
            {
                Metric.MacroAccuracy => value.MacroAccuracy.Mean,
                Metric.MicroAccuracy => value.MicroAccuracy.Mean,
                Metric.LogLoss => value.LogLoss.Mean,
                _ => 0.0
            };
            if (metricValue == 0.0f)
                continue;
            var importance = metricValue * -1;
            var valueString = importance.ToString(
                CultureInfo.InvariantCulture);
            sb.AppendLine(
                $"{key}, {valueString}");
        }

        return sb.ToString();
    }
}