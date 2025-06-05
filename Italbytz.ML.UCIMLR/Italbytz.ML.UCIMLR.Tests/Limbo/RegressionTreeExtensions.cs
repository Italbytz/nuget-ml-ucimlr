using System.Text;
using Microsoft.ML.Trainers.FastTree;

namespace Italbytz.ML.Trainers.FastTree;

public static class RegressionTreeExtensions
{
    public static string ToGraphviz(
        this RegressionTree tree)
    {
        var sb = new StringBuilder();
        sb.AppendLine("digraph G {");
        sb.AppendLine("    rankdir=\"TB\"");
        for (var i = 0; i < tree.LeafValues.Count; i++)
            sb.AppendLine(
                $"    l{i} [shape=box,label={tree.LeafValues[i]:F2}];");
        for (var i = 0; i < tree.NumericalSplitFeatureIndexes.Count; i++)
        {
            var featureIndex = tree.NumericalSplitFeatureIndexes[i];
            var threshold = tree.NumericalSplitThresholds[i];
            sb.AppendLine(
                $"    n{i} [shape=plain,label=<Feature{featureIndex}<br/>{threshold:F2}>];");
        }

        for (var i = 0; i < tree.LeftChild.Count; i++)
        {
            var leftChildType = tree.LeftChild[i] < 0 ? "l" : "n";
            var leftChildIndex = tree.LeftChild[i] < 0
                ? ~tree.LeftChild[i]
                : tree.LeftChild[i];
            var rightChildType = tree.RightChild[i] < 0 ? "l" : "n";
            var rightChildIndex = tree.RightChild[i] < 0
                ? ~tree.RightChild[i]
                : tree.RightChild[i];
            var leftChild = leftChildType + leftChildIndex;
            var rightChild = rightChildType + rightChildIndex;
            sb.AppendLine($"    n{i} -> {leftChild} [label=\"≤\"];");
            sb.AppendLine($"    n{i} -> n{rightChild} [label=\">\";]");
        }

        sb.AppendLine("}");
        return sb.ToString();
    }

    public static string ToPlantUML(
        this RegressionTree tree)
    {
        var sb = new StringBuilder();
        sb.AppendLine("@startuml");
        sb.AppendLine("object RegressionTree {");
        sb.AppendLine(
            $"    int[] NumericalSplitFeatureIndexes = [{string.Join(", ", tree.NumericalSplitFeatureIndexes)}]");
        sb.AppendLine(
            $"    double[] NumericalSplitThresholds = [{string.Join(", ", tree.NumericalSplitThresholds)}]");
        sb.AppendLine(
            $"    int[] LeftChild = [{string.Join(", ", tree.LeftChild)}]");
        sb.AppendLine(
            $"    int[] RightChild = [{string.Join(", ", tree.RightChild)}]");
        sb.AppendLine(
            $"    double[] LeafValues = [{string.Join(", ", tree.LeafValues)}]");
        sb.AppendLine("}");
        sb.AppendLine("@enduml");
        return sb.ToString();
    }
}