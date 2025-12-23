//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1};
use sklears_core::error::{Result as SklResult, SklearsError};
use std::collections::HashMap;
type Result<T> = SklResult<T>;
type Float = f64;
/// Complex return type for bioinformatics analysis methods
/// (scores, p_values, fold_changes, metrics, additional_data)
type BioinformaticsAnalysisResult = (
    Array1<Float>,
    Option<Array1<Float>>,
    Option<Array1<Float>>,
    Option<HashMap<String, Float>>,
    Option<Array1<Float>>,
);
pub(crate) fn apply_normalization(x: &Array2<Float>, method: &str) -> Result<Array2<Float>> {
    match method {
        "log2" => {
            let normalized = x.mapv(|val| if val > 0.0 { (val + 1.0).log2() } else { 0.0 });
            Ok(normalized)
        }
        "quantile" => {
            let mut normalized = x.clone();
            let (n_samples, n_features) = x.dim();
            for j in 0..n_features {
                let mut column: Vec<Float> = x.column(j).to_vec();
                column.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median = if column.len() % 2 == 0 {
                    (column[column.len() / 2 - 1] + column[column.len() / 2]) / 2.0
                } else {
                    column[column.len() / 2]
                };
                for i in 0..n_samples {
                    normalized[[i, j]] -= median;
                }
            }
            Ok(normalized)
        }
        "rma" => apply_normalization(x, "quantile"),
        _ => Ok(x.clone()),
    }
}
pub(crate) fn apply_multiple_testing_correction(
    p_values: &Array1<Float>,
    method: &str,
) -> Result<Array1<Float>> {
    let n = p_values.len();
    let mut adjusted = p_values.clone();
    match method {
        "fdr" => {
            let mut indexed_p: Vec<(usize, Float)> =
                p_values.iter().enumerate().map(|(i, &p)| (i, p)).collect();
            indexed_p.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            for (rank, &(original_idx, p_val)) in indexed_p.iter().enumerate() {
                let corrected_p = p_val * (n as Float) / ((rank + 1) as Float);
                adjusted[original_idx] = corrected_p.min(1.0);
            }
            for i in (0..indexed_p.len() - 1).rev() {
                let curr_idx = indexed_p[i].0;
                let next_idx = indexed_p[i + 1].0;
                adjusted[curr_idx] = adjusted[curr_idx].min(adjusted[next_idx]);
            }
        }
        "bonferroni" => {
            adjusted = p_values.mapv(|p| (p * n as Float).min(1.0));
        }
        "holm" => {
            let mut indexed_p: Vec<(usize, Float)> =
                p_values.iter().enumerate().map(|(i, &p)| (i, p)).collect();
            indexed_p.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            for (rank, &(original_idx, p_val)) in indexed_p.iter().enumerate() {
                let corrected_p = p_val * ((n - rank) as Float);
                adjusted[original_idx] = corrected_p.min(1.0);
            }
        }
        "none" => {}
        _ => {
            return Err(SklearsError::InvalidInput(format!(
                "Unknown multiple testing correction method: {}",
                method
            )));
        }
    }
    Ok(adjusted)
}
pub(crate) fn separate_groups(
    feature: &ArrayView1<Float>,
    y: &Array1<Float>,
) -> (Vec<Float>, Vec<Float>) {
    let mut group0 = Vec::new();
    let mut group1 = Vec::new();
    for i in 0..feature.len() {
        if y[i] == 0.0 {
            group0.push(feature[i]);
        } else {
            group1.push(feature[i]);
        }
    }
    (group0, group1)
}
pub(crate) fn compute_mean(values: &[Float]) -> Float {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<Float>() / values.len() as Float
    }
}
pub(crate) fn compute_std(values: &[Float], mean: Float) -> Float {
    if values.len() <= 1 {
        1.0
    } else {
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<Float>() / (values.len() - 1) as Float;
        variance.sqrt()
    }
}
pub(crate) fn compute_pathway_scores(
    gene_scores: &Array1<Float>,
    database: &str,
) -> Result<HashMap<String, Float>> {
    let mut pathway_scores = HashMap::new();
    let len = gene_scores.len();
    let sum_slice = |start: usize, end: usize| -> Float {
        if start >= len {
            0.0
        } else {
            let actual_end = end.min(len);
            if actual_end <= start {
                0.0
            } else {
                gene_scores.slice(s![start..actual_end]).sum()
            }
        }
    };
    match database {
        "kegg" => {
            pathway_scores.insert("metabolism".to_string(), sum_slice(0, 100));
            pathway_scores.insert("signaling".to_string(), sum_slice(100, 200));
            pathway_scores.insert("immune".to_string(), sum_slice(200, 300));
        }
        "go" => {
            pathway_scores.insert("biological_process".to_string(), gene_scores.sum() * 0.4);
            pathway_scores.insert("molecular_function".to_string(), gene_scores.sum() * 0.3);
            pathway_scores.insert("cellular_component".to_string(), gene_scores.sum() * 0.3);
        }
        "reactome" => {
            pathway_scores.insert("dna_repair".to_string(), sum_slice(0, 50));
            pathway_scores.insert("cell_cycle".to_string(), sum_slice(50, 100));
        }
        _ => {
            pathway_scores.insert("custom_pathway".to_string(), gene_scores.sum());
        }
    }
    Ok(pathway_scores)
}
pub(crate) fn map_pathway_scores_to_genes(
    gene_scores: &Array1<Float>,
    pathway_scores: &HashMap<String, Float>,
) -> Result<Array1<Float>> {
    let n_genes = gene_scores.len();
    let mut biological_scores = Array1::zeros(n_genes);
    let max_pathway_score = pathway_scores.values().fold(0.0_f64, |a, &b| a.max(b));
    for i in 0..n_genes {
        if i < n_genes / 3 {
            biological_scores[i] =
                pathway_scores.get("metabolism").unwrap_or(&0.0) / max_pathway_score;
        } else if i < 2 * n_genes / 3 {
            biological_scores[i] =
                pathway_scores.get("signaling").unwrap_or(&0.0) / max_pathway_score;
        } else {
            biological_scores[i] = pathway_scores.get("immune").unwrap_or(&0.0) / max_pathway_score;
        }
    }
    Ok(biological_scores)
}
pub(crate) fn compute_minor_allele_frequency(snp: &ArrayView1<Float>) -> Float {
    let mut allele_counts = [0, 0, 0];
    for &genotype in snp.iter() {
        let rounded = genotype.round() as i32;
        if (0..=2).contains(&rounded) {
            allele_counts[rounded as usize] += 1;
        }
    }
    let total_alleles = 2 * (allele_counts[0] + allele_counts[1] + allele_counts[2]);
    if total_alleles == 0 {
        return 0.0;
    }
    let minor_allele_count = allele_counts[1] + 2 * allele_counts[2];
    let major_allele_count = total_alleles - minor_allele_count;
    (minor_allele_count.min(major_allele_count) as Float) / (total_alleles as Float)
}
pub(crate) fn compute_hardy_weinberg_p_value(snp: &ArrayView1<Float>) -> Float {
    let mut genotype_counts = [0, 0, 0];
    for &genotype in snp.iter() {
        let rounded = genotype.round() as i32;
        if (0..=2).contains(&rounded) {
            genotype_counts[rounded as usize] += 1;
        }
    }
    let total = genotype_counts[0] + genotype_counts[1] + genotype_counts[2];
    if total == 0 {
        return 1.0;
    }
    let p = ((2 * genotype_counts[0] + genotype_counts[1]) as Float) / (2.0 * total as Float);
    let q = 1.0 - p;
    let expected_aa = p * p * total as Float;
    let expected_ab = 2.0 * p * q * total as Float;
    let expected_bb = q * q * total as Float;
    let chi2 = ((genotype_counts[0] as Float - expected_aa).powi(2) / expected_aa)
        + ((genotype_counts[1] as Float - expected_ab).powi(2) / expected_ab)
        + ((genotype_counts[2] as Float - expected_bb).powi(2) / expected_bb);
    chi_square_to_p_value(chi2, 1)
}
pub(crate) fn compute_chi_square_association(
    snp: &ArrayView1<Float>,
    phenotype: &Array1<Float>,
) -> Float {
    let mut contingency = [[0, 0, 0], [0, 0, 0]];
    for i in 0..snp.len() {
        let genotype = snp[i].round() as usize;
        let pheno = phenotype[i].round() as usize;
        if genotype <= 2 && pheno <= 1 {
            contingency[pheno][genotype] += 1;
        }
    }
    let mut chi2 = 0.0;
    let total = contingency.iter().flatten().sum::<i32>() as Float;
    if total == 0.0 {
        return 0.0;
    }
    for i in 0..2 {
        for j in 0..3 {
            let observed = contingency[i][j] as Float;
            let row_sum = contingency[i].iter().sum::<i32>() as Float;
            let col_sum = (contingency[0][j] + contingency[1][j]) as Float;
            let expected = (row_sum * col_sum) / total;
            if expected > 0.0 {
                chi2 += (observed - expected).powi(2) / expected;
            }
        }
    }
    chi2
}
pub(crate) fn chi_square_to_p_value(chi2: Float, df: i32) -> Float {
    (1.0 + chi2 / (df as Float + 1.0)).recip()
}
pub(crate) fn correct_population_structure(p_values: &Array1<Float>) -> Result<Array1<Float>> {
    let mut sorted_p: Vec<Float> = p_values.to_vec();
    sorted_p.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_chi2 = -2.0 * sorted_p[sorted_p.len() / 2].ln();
    let lambda = median_chi2 / 0.693;
    let corrected = p_values.mapv(|p| {
        let chi2 = -2.0 * p.ln();
        let corrected_chi2 = chi2 / lambda;
        (-corrected_chi2 / 2.0).exp()
    });
    Ok(corrected)
}
pub(crate) fn compute_linkage_disequilibrium_r2(
    snp1: &ArrayView1<Float>,
    snp2: &ArrayView1<Float>,
) -> Float {
    let correlation = compute_pearson_correlation(snp1, snp2);
    correlation * correlation
}
pub(crate) fn compute_protein_network_centrality(x: &Array2<Float>) -> Result<Array1<Float>> {
    let (_, n_features) = x.dim();
    let mut centrality = Array1::zeros(n_features);
    for i in 0..n_features {
        let feature_i = x.column(i);
        let mut degree = 0.0;
        for j in 0..n_features {
            if i != j {
                let feature_j = x.column(j);
                let correlation = compute_pearson_correlation(&feature_i, &feature_j);
                if correlation.abs() > 0.3 {
                    degree += 1.0;
                }
            }
        }
        centrality[i] = degree / (n_features - 1) as Float;
    }
    Ok(centrality)
}
pub(crate) fn compute_functional_domain_scores(x: &Array2<Float>) -> Result<Array1<Float>> {
    let (_, n_features) = x.dim();
    let mut scores = Array1::zeros(n_features);
    for i in 0..n_features {
        scores[i] = if i % 10 == 0 { 1.0 } else { 0.5 };
    }
    Ok(scores)
}
pub(crate) fn compute_pearson_correlation(x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Float {
    let n = x.len();
    if n != y.len() || n == 0 {
        return 0.0;
    }
    let mean_x = x.sum() / n as Float;
    let mean_y = y.sum() / n as Float;
    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;
    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        numerator += dx * dy;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
    }
    let denominator = (sum_sq_x * sum_sq_y).sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}
pub(crate) fn compute_size_factor(counts: &[Float]) -> Float {
    let geometric_mean = if counts.is_empty() {
        1.0
    } else {
        let log_sum: Float = counts.iter().map(|&x| (x + 1.0).ln()).sum();
        (log_sum / counts.len() as Float).exp()
    };
    if geometric_mean < 1e-10 {
        1.0
    } else {
        geometric_mean
    }
}
pub(crate) fn estimate_dispersion(
    counts0: &[Float],
    counts1: &[Float],
    mean0: Float,
    mean1: Float,
) -> Float {
    let var0 = if counts0.len() > 1 {
        compute_variance(counts0, mean0)
    } else {
        mean0
    };
    let var1 = if counts1.len() > 1 {
        compute_variance(counts1, mean1)
    } else {
        mean1
    };
    let dispersion0 = if mean0 > 1e-10 {
        ((var0 - mean0) / (mean0 * mean0)).max(0.01)
    } else {
        0.1
    };
    let dispersion1 = if mean1 > 1e-10 {
        ((var1 - mean1) / (mean1 * mean1)).max(0.01)
    } else {
        0.1
    };
    (dispersion0 + dispersion1) / 2.0
}
pub(crate) fn compute_wald_standard_error(
    mean0: Float,
    mean1: Float,
    dispersion: Float,
    n0: usize,
    n1: usize,
) -> Float {
    let var0 = mean0 * (1.0 + dispersion * mean0);
    let var1 = mean1 * (1.0 + dispersion * mean1);
    let se_squared = var0 / (n0 as Float) + var1 / (n1 as Float);
    if se_squared > 0.0 {
        se_squared.sqrt() / ((mean0 + mean1) / 2.0 + 1e-10)
    } else {
        1.0
    }
}
pub(crate) fn normal_cdf(x: Float) -> Float {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989423 * (-x * x / 2.0).exp();
    let prob =
        d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    if x >= 0.0 {
        1.0 - prob
    } else {
        prob
    }
}
pub(crate) fn separate_groups_indices(y: &Array1<Float>) -> (Vec<usize>, Vec<usize>) {
    let mut group0 = Vec::new();
    let mut group1 = Vec::new();
    for (i, &val) in y.iter().enumerate() {
        if val == 0.0 {
            group0.push(i);
        } else {
            group1.push(i);
        }
    }
    (group0, group1)
}
pub(crate) fn estimate_common_dispersion(counts0: &[Float], counts1: &[Float]) -> Float {
    let mean0 = compute_mean(counts0);
    let mean1 = compute_mean(counts1);
    let overall_mean = (mean0 + mean1) / 2.0;
    if overall_mean < 1e-10 {
        return 0.1;
    }
    let var0 = compute_variance(counts0, mean0);
    let var1 = compute_variance(counts1, mean1);
    let overall_var = (var0 + var1) / 2.0;
    ((overall_var - overall_mean) / (overall_mean * overall_mean)).max(0.01)
}
pub(crate) fn compute_exact_test_p_value(
    count0: Float,
    _count1: Float,
    total: Float,
    p0: Float,
    _dispersion: Float,
) -> Float {
    if total < 1e-10 {
        return 1.0;
    }
    let observed_proportion = count0 / total;
    let expected_proportion = p0;
    let se = (p0 * (1.0 - p0) / total).sqrt();
    if se < 1e-10 {
        return 1.0;
    }
    let z = (observed_proportion - expected_proportion).abs() / se;
    2.0 * (1.0 - normal_cdf(z))
}
pub(crate) fn compute_variance(values: &[Float], mean: Float) -> Float {
    if values.len() <= 1 {
        return mean.max(1.0);
    }
    let sum_sq: Float = values.iter().map(|&x| (x - mean).powi(2)).sum();
    sum_sq / (values.len() - 1) as Float
}
pub(crate) fn estimate_prior_variance(gene_variances: &[Float]) -> (Float, Float) {
    if gene_variances.is_empty() {
        return (1.0, 4.0);
    }
    let mut sorted_vars = gene_variances.to_vec();
    sorted_vars.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let s0_squared = if sorted_vars.len() % 2 == 0 {
        (sorted_vars[sorted_vars.len() / 2 - 1] + sorted_vars[sorted_vars.len() / 2]) / 2.0
    } else {
        sorted_vars[sorted_vars.len() / 2]
    };
    let d0 = 4.0;
    (s0_squared.max(0.01), d0)
}
pub(crate) fn define_gene_sets(n_genes: usize) -> HashMap<String, Vec<usize>> {
    let mut gene_sets = HashMap::new();
    let set_size = (n_genes / 10).max(5);
    let metabolism: Vec<usize> = (0..set_size.min(n_genes)).collect();
    gene_sets.insert("metabolism".to_string(), metabolism);
    let signaling: Vec<usize> = (set_size..((2 * set_size).min(n_genes))).collect();
    gene_sets.insert("signaling".to_string(), signaling);
    let immune: Vec<usize> = ((2 * set_size)..((3 * set_size).min(n_genes))).collect();
    gene_sets.insert("immune_response".to_string(), immune);
    let cell_cycle: Vec<usize> = ((3 * set_size)..((4 * set_size).min(n_genes))).collect();
    gene_sets.insert("cell_cycle".to_string(), cell_cycle);
    let apoptosis: Vec<usize> = ((4 * set_size)..((5 * set_size).min(n_genes))).collect();
    gene_sets.insert("apoptosis".to_string(), apoptosis);
    gene_sets
}
pub(crate) fn compute_enrichment_score(
    ranked_genes: &[(usize, Float)],
    gene_set: &[usize],
) -> Float {
    if gene_set.is_empty() || ranked_genes.is_empty() {
        return 0.0;
    }
    let n = ranked_genes.len() as Float;
    let n_set = gene_set.len() as Float;
    let mut max_es = 0.0;
    let mut current_es = 0.0;
    let sum_corr: Float = ranked_genes
        .iter()
        .filter(|(idx, _)| gene_set.contains(idx))
        .map(|(_, score)| score.abs())
        .sum();
    if sum_corr < 1e-10 {
        return 0.0;
    }
    for (idx, score) in ranked_genes.iter() {
        if gene_set.contains(idx) {
            current_es += score.abs() / sum_corr;
        } else {
            current_es -= 1.0 / (n - n_set);
        }
        max_es = (max_es as Float).max(current_es.abs());
    }
    max_es
}
pub(crate) fn compute_hypergeometric_p_value(
    overlap: usize,
    gene_set_size: usize,
    background_size: usize,
    sig_gene_count: usize,
) -> Float {
    if sig_gene_count == 0 || gene_set_size == 0 {
        return 1.0;
    }
    let total = gene_set_size + background_size;
    let expected = (gene_set_size as Float * sig_gene_count as Float) / total as Float;
    if expected < 1e-10 {
        return 1.0;
    }
    let variance = expected
        * (1.0 - gene_set_size as Float / total as Float)
        * ((total - sig_gene_count) as Float / (total - 1) as Float);
    let se = variance.sqrt();
    if se < 1e-10 {
        return 1.0;
    }
    let z = ((overlap as Float) - expected) / se;
    if z > 0.0 {
        1.0 - normal_cdf(z)
    } else {
        normal_cdf(z)
    }
}
pub(crate) fn build_ppi_network(x: &Array2<Float>, correlation_threshold: Float) -> Array2<bool> {
    let n_proteins = x.ncols();
    let mut network = Array2::from_elem((n_proteins, n_proteins), false);
    for i in 0..n_proteins {
        for j in (i + 1)..n_proteins {
            let protein_i = x.column(i);
            let protein_j = x.column(j);
            let correlation = compute_pearson_correlation(&protein_i, &protein_j);
            if correlation.abs() > correlation_threshold {
                network[[i, j]] = true;
                network[[j, i]] = true;
            }
        }
    }
    network
}
pub(crate) fn compute_degree_centrality(network: &Array2<bool>) -> Array1<Float> {
    let n = network.nrows();
    let mut centrality = Array1::zeros(n);
    for i in 0..n {
        let degree: usize = network.row(i).iter().filter(|&&x| x).count();
        centrality[i] = degree as Float / (n - 1) as Float;
    }
    centrality
}
pub(crate) fn compute_betweenness_centrality(network: &Array2<bool>) -> Array1<Float> {
    let n = network.nrows();
    let mut centrality = Array1::zeros(n);
    for s in 0..n {
        for t in (s + 1)..n {
            let paths = bfs_shortest_paths(network, s, t);
            for &node in &paths {
                if node != s && node != t {
                    centrality[node] += 1.0;
                }
            }
        }
    }
    let max_centrality = centrality.iter().fold(0.0_f64, |a, &b| a.max(b));
    if max_centrality > 1e-10 {
        centrality.mapv_inplace(|x| x / max_centrality);
    }
    centrality
}
pub(crate) fn bfs_shortest_paths(network: &Array2<bool>, start: usize, end: usize) -> Vec<usize> {
    let n = network.nrows();
    let mut visited = vec![false; n];
    let mut parent = vec![None; n];
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(start);
    visited[start] = true;
    while let Some(node) = queue.pop_front() {
        if node == end {
            break;
        }
        for neighbor in 0..n {
            if network[[node, neighbor]] && !visited[neighbor] {
                visited[neighbor] = true;
                parent[neighbor] = Some(node);
                queue.push_back(neighbor);
            }
        }
    }
    let mut path = Vec::new();
    let mut current = Some(end);
    while let Some(node) = current {
        path.push(node);
        current = parent[node];
    }
    path.reverse();
    path
}
pub(crate) fn compute_closeness_centrality(network: &Array2<bool>) -> Array1<Float> {
    let n = network.nrows();
    let mut centrality = Array1::zeros(n);
    for i in 0..n {
        let mut total_distance = 0.0;
        let mut reachable = 0;
        for j in 0..n {
            if i != j {
                let dist = compute_shortest_path_length(network, i, j);
                if dist < Float::INFINITY {
                    total_distance += dist;
                    reachable += 1;
                }
            }
        }
        if reachable > 0 && total_distance > 1e-10 {
            centrality[i] = (reachable as Float) / total_distance;
        }
    }
    let max_centrality = centrality.iter().fold(0.0_f64, |a, &b| a.max(b));
    if max_centrality > 1e-10 {
        centrality.mapv_inplace(|x| x / max_centrality);
    }
    centrality
}
pub(crate) fn compute_shortest_path_length(
    network: &Array2<bool>,
    start: usize,
    end: usize,
) -> Float {
    let n = network.nrows();
    let mut visited = vec![false; n];
    let mut distance = vec![Float::INFINITY; n];
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(start);
    visited[start] = true;
    distance[start] = 0.0;
    while let Some(node) = queue.pop_front() {
        if node == end {
            return distance[end];
        }
        for neighbor in 0..n {
            if network[[node, neighbor]] && !visited[neighbor] {
                visited[neighbor] = true;
                distance[neighbor] = distance[node] + 1.0;
                queue.push_back(neighbor);
            }
        }
    }
    distance[end]
}
pub(crate) fn compute_protein_impact_score(variant: &ArrayView1<Float>, position: usize) -> Float {
    let maf = compute_minor_allele_frequency(variant);
    let rarity_score = if maf < 0.01 { 1.0 - maf / 0.01 } else { 0.5 };
    let position_score = ((position % 100) as Float / 100.0).sin().abs();
    (rarity_score + position_score) / 2.0
}
pub(crate) fn compute_variant_domain_score(position: usize, total_features: usize) -> Float {
    let relative_pos = (position as Float) / (total_features as Float);
    if relative_pos < 0.2 || (relative_pos > 0.4 && relative_pos < 0.6) || relative_pos > 0.8 {
        0.8
    } else {
        0.3
    }
}
pub(crate) fn compute_layer_correlation(
    layer1: &Array2<Float>,
    layer2: &Array2<Float>,
) -> Array2<Float> {
    let n1 = layer1.ncols();
    let n2 = layer2.ncols();
    let mut correlation_matrix = Array2::zeros((n1, n2));
    for i in 0..n1 {
        for j in 0..n2 {
            let feature1 = layer1.column(i);
            let feature2 = layer2.column(j);
            let correlation = compute_pearson_correlation(&feature1, &feature2);
            correlation_matrix[[i, j]] = correlation;
        }
    }
    correlation_matrix
}
#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::super::types::BioinformaticsFeatureSelector;
    use super::*;
    use crate::base::SelectorMixin;
    use scirs2_core::ndarray::{Array1, Array2};
    use sklears_core::traits::{Fit, Transform};
    #[test]
    fn test_bioinformatics_feature_selector_creation() {
        let selector = BioinformaticsFeatureSelector::new();
        assert_eq!(selector.data_type, "gene_expression");
        assert_eq!(selector.analysis_method, "differential_expression");
        assert_eq!(selector.multiple_testing_correction, "fdr");
    }
    #[test]
    fn test_bioinformatics_feature_selector_builder() {
        let selector = BioinformaticsFeatureSelector::builder()
            .data_type("snp")
            .analysis_method("association_test")
            .significance_threshold(0.01)
            .k(100)
            .build();
        assert_eq!(selector.data_type, "snp");
        assert_eq!(selector.analysis_method, "association_test");
        assert_eq!(selector.significance_threshold, 0.01);
        assert_eq!(selector.k, 100);
    }
    #[test]
    fn test_differential_expression_analysis() {
        let expression_data = Array2::from_shape_vec(
            (6, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 6.0, 10.0, 11.0, 12.0, 13.0,
                11.0, 12.0, 13.0, 14.0, 12.0, 13.0, 14.0, 15.0,
            ],
        )
        .unwrap();
        let labels = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let selector = BioinformaticsFeatureSelector::builder()
            .data_type("gene_expression")
            .analysis_method("differential_expression")
            .k(2)
            .build();
        let trained = selector.fit(&expression_data, &labels).unwrap();
        let transformed = trained.transform(&expression_data).unwrap();
        assert_eq!(transformed.ncols(), 2);
        assert_eq!(transformed.nrows(), 6);
    }
    #[test]
    fn test_apply_normalization() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 4.0, 2.0, 8.0, 4.0, 16.0]).unwrap();
        let log_normalized = apply_normalization(&data, "log2").unwrap();
        assert_eq!(log_normalized.dim(), (3, 2));
        assert!((log_normalized[[0, 0]] - (2.0_f64).log2()).abs() < 1e-6);
        assert!((log_normalized[[0, 1]] - (5.0_f64).log2()).abs() < 1e-6);
    }
    #[test]
    fn test_multiple_testing_correction() {
        let p_values = Array1::from_vec(vec![0.01, 0.05, 0.1, 0.2, 0.5]);
        let fdr_corrected = apply_multiple_testing_correction(&p_values, "fdr").unwrap();
        assert_eq!(fdr_corrected.len(), 5);
        let bonferroni_corrected =
            apply_multiple_testing_correction(&p_values, "bonferroni").unwrap();
        assert_eq!(bonferroni_corrected.len(), 5);
        assert!(bonferroni_corrected[0] >= p_values[0]);
    }
    #[test]
    fn test_minor_allele_frequency() {
        let snp = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
        let maf = compute_minor_allele_frequency(&snp.view());
        assert!((maf - 0.5).abs() < 1e-6);
    }
    #[test]
    fn test_hardy_weinberg_test() {
        let snp = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
        let hwe_p = compute_hardy_weinberg_p_value(&snp.view());
        assert!(hwe_p >= 0.0 && hwe_p <= 1.0);
    }
    #[test]
    fn test_get_support() {
        let expression_data = Array2::from_shape_vec(
            (4, 6),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0, 11.0, 12.0, 13.0,
                14.0, 15.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            ],
        )
        .unwrap();
        let labels = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let selector = BioinformaticsFeatureSelector::builder().k(3).build();
        let trained = selector.fit(&expression_data, &labels).unwrap();
        let support = trained.get_support().unwrap();
        assert_eq!(support.len(), 6);
        assert_eq!(support.iter().filter(|&&x| x).count(), 3);
    }
    #[test]
    fn test_snp_association_analysis() {
        let snp_data = Array2::from_shape_vec(
            (6, 3),
            vec![
                0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0,
                0.0, 1.0,
            ],
        )
        .unwrap();
        let phenotype = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let selector = BioinformaticsFeatureSelector::builder()
            .data_type("snp")
            .analysis_method("association_test")
            .k(2)
            .build();
        let trained = selector.fit(&snp_data, &phenotype).unwrap();
        let transformed = trained.transform(&snp_data).unwrap();
        assert_eq!(transformed.ncols(), 2);
        assert_eq!(transformed.nrows(), 6);
    }
}
