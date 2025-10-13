use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::api_data_structures::{TraitInfo, TypeInfo};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TutorialSystem {
    pub tutorials: Vec<Tutorial>,
    pub learning_paths: Vec<LearningPath>,
    pub progress_tracker: ProgressTracker,
    pub assessment_engine: AssessmentEngine,
    pub config: TutorialConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TutorialConfig {
    pub difficulty_levels: Vec<DifficultyLevel>,
    pub enable_interactive_examples: bool,
    pub enable_progress_tracking: bool,
    pub enable_assessments: bool,
    pub max_tutorial_duration: u32,
    pub code_execution_timeout: u32,
    pub personalization_enabled: bool,
}

impl Default for TutorialConfig {
    fn default() -> Self {
        Self {
            difficulty_levels: vec![
                DifficultyLevel::Beginner,
                DifficultyLevel::Intermediate,
                DifficultyLevel::Advanced,
            ],
            enable_interactive_examples: true,
            enable_progress_tracking: true,
            enable_assessments: true,
            max_tutorial_duration: 3600,
            code_execution_timeout: 30,
            personalization_enabled: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DifficultyLevel {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tutorial {
    pub id: String,
    pub title: String,
    pub description: String,
    pub difficulty: DifficultyLevel,
    pub duration_minutes: u32,
    pub prerequisites: Vec<String>,
    pub learning_objectives: Vec<String>,
    pub sections: Vec<TutorialSection>,
    pub assessment: Option<Assessment>,
    pub metadata: TutorialMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TutorialMetadata {
    pub author: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub version: String,
    pub tags: Vec<String>,
    pub category: TutorialCategory,
    pub language: String,
    pub popularity_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TutorialCategory {
    GettingStarted,
    CoreConcepts,
    AdvancedFeatures,
    BestPractices,
    RealWorldExamples,
    Performance,
    Testing,
    Integration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TutorialSection {
    pub id: String,
    pub title: String,
    pub content: SectionContent,
    pub interactive_elements: Vec<InteractiveElement>,
    pub estimated_duration: u32,
    pub completion_criteria: CompletionCriteria,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SectionContent {
    Text {
        content: String,
        format: ContentFormat,
    },
    Code {
        content: String,
        language: String,
        runnable: bool,
    },
    Exercise {
        description: String,
        starter_code: String,
        solution: String,
    },
    Quiz {
        questions: Vec<QuizQuestion>,
    },
    Video {
        url: String,
        duration: u32,
        transcript: Option<String>,
    },
    Interactive {
        component_type: String,
        config: serde_json::Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentFormat {
    Markdown,
    Html,
    PlainText,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractiveElement {
    pub element_type: InteractiveElementType,
    pub config: serde_json::Value,
    pub validation_rules: Vec<ValidationRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractiveElementType {
    CodeEditor,
    LiveExample,
    Quiz,
    Diagram,
    Simulation,
    Playground,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    pub rule_type: ValidationType,
    pub condition: String,
    pub error_message: String,
    pub hint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationType {
    Compilation,
    Runtime,
    Output,
    Style,
    Performance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionCriteria {
    pub required_interactions: Vec<String>,
    pub minimum_score: Option<f64>,
    pub time_spent_minimum: Option<u32>,
    pub code_execution_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningPath {
    pub id: String,
    pub title: String,
    pub description: String,
    pub difficulty: DifficultyLevel,
    pub estimated_hours: u32,
    pub tutorial_sequence: Vec<String>,
    pub completion_rewards: Vec<String>,
    pub prerequisites: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressTracker {
    pub user_progress: HashMap<String, UserProgress>,
    pub global_statistics: GlobalStatistics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProgress {
    pub user_id: String,
    pub completed_tutorials: HashSet<String>,
    pub current_tutorial: Option<String>,
    pub current_section: Option<String>,
    pub completion_percentage: f64,
    pub time_spent: u32,
    pub assessment_scores: HashMap<String, f64>,
    pub achievements: Vec<Achievement>,
    pub learning_preferences: LearningPreferences,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalStatistics {
    pub total_users: usize,
    pub tutorial_completion_rates: HashMap<String, f64>,
    pub average_scores: HashMap<String, f64>,
    pub popular_tutorials: Vec<String>,
    pub common_challenges: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Achievement {
    pub id: String,
    pub title: String,
    pub description: String,
    pub icon: String,
    pub earned_at: chrono::DateTime<chrono::Utc>,
    pub rarity: AchievementRarity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AchievementRarity {
    Common,
    Uncommon,
    Rare,
    Epic,
    Legendary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningPreferences {
    pub preferred_difficulty: DifficultyLevel,
    pub learning_pace: LearningPace,
    pub content_types: Vec<ContentType>,
    pub reminder_frequency: ReminderFrequency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningPace {
    Slow,
    Normal,
    Fast,
    SelfPaced,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentType {
    Text,
    Video,
    Interactive,
    Code,
    Exercises,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReminderFrequency {
    Never,
    Daily,
    Weekly,
    Monthly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssessmentEngine {
    pub assessments: HashMap<String, Assessment>,
    pub question_bank: Vec<QuizQuestion>,
    pub scoring_algorithms: HashMap<String, ScoringAlgorithm>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assessment {
    pub id: String,
    pub title: String,
    pub description: String,
    pub questions: Vec<String>,
    pub time_limit: Option<u32>,
    pub passing_score: f64,
    pub max_attempts: Option<u32>,
    pub feedback_mode: FeedbackMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackMode {
    Immediate,
    EndOfAssessment,
    Manual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuizQuestion {
    pub id: String,
    pub question_type: QuestionType,
    pub content: String,
    pub options: Vec<String>,
    pub correct_answer: String,
    pub explanation: String,
    pub difficulty: DifficultyLevel,
    pub tags: Vec<String>,
    pub points: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuestionType {
    MultipleChoice,
    TrueFalse,
    ShortAnswer,
    CodeCompletion,
    CodeReview,
    Matching,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringAlgorithm {
    pub name: String,
    pub description: String,
    pub formula: String,
    pub parameters: HashMap<String, f64>,
}

pub struct TutorialBuilder {
    tutorial: Tutorial,
    current_section: Option<TutorialSection>,
}

impl TutorialBuilder {
    pub fn new(id: String, title: String) -> Self {
        Self {
            tutorial: Tutorial {
                id,
                title,
                description: String::new(),
                difficulty: DifficultyLevel::Beginner,
                duration_minutes: 0,
                prerequisites: Vec::new(),
                learning_objectives: Vec::new(),
                sections: Vec::new(),
                assessment: None,
                metadata: TutorialMetadata {
                    author: String::new(),
                    created_at: chrono::Utc::now(),
                    updated_at: chrono::Utc::now(),
                    version: "1.0.0".to_string(),
                    tags: Vec::new(),
                    category: TutorialCategory::GettingStarted,
                    language: "en".to_string(),
                    popularity_score: 0.0,
                },
            },
            current_section: None,
        }
    }

    pub fn description(mut self, description: String) -> Self {
        self.tutorial.description = description;
        self
    }

    pub fn difficulty(mut self, difficulty: DifficultyLevel) -> Self {
        self.tutorial.difficulty = difficulty;
        self
    }

    pub fn duration(mut self, minutes: u32) -> Self {
        self.tutorial.duration_minutes = minutes;
        self
    }

    pub fn prerequisite(mut self, prerequisite: String) -> Self {
        self.tutorial.prerequisites.push(prerequisite);
        self
    }

    pub fn learning_objective(mut self, objective: String) -> Self {
        self.tutorial.learning_objectives.push(objective);
        self
    }

    pub fn author(mut self, author: String) -> Self {
        self.tutorial.metadata.author = author;
        self
    }

    pub fn category(mut self, category: TutorialCategory) -> Self {
        self.tutorial.metadata.category = category;
        self
    }

    pub fn tag(mut self, tag: String) -> Self {
        self.tutorial.metadata.tags.push(tag);
        self
    }

    pub fn section(mut self, id: String, title: String) -> Self {
        if let Some(section) = self.current_section.take() {
            self.tutorial.sections.push(section);
        }

        self.current_section = Some(TutorialSection {
            id,
            title,
            content: SectionContent::Text {
                content: String::new(),
                format: ContentFormat::Markdown,
            },
            interactive_elements: Vec::new(),
            estimated_duration: 0,
            completion_criteria: CompletionCriteria {
                required_interactions: Vec::new(),
                minimum_score: None,
                time_spent_minimum: None,
                code_execution_required: false,
            },
        });

        self
    }

    pub fn text_content(mut self, content: String, format: ContentFormat) -> Self {
        if let Some(ref mut section) = self.current_section {
            section.content = SectionContent::Text { content, format };
        }
        self
    }

    pub fn code_content(mut self, content: String, language: String, runnable: bool) -> Self {
        if let Some(ref mut section) = self.current_section {
            section.content = SectionContent::Code {
                content,
                language,
                runnable,
            };
        }
        self
    }

    pub fn exercise(mut self, description: String, starter_code: String, solution: String) -> Self {
        if let Some(ref mut section) = self.current_section {
            section.content = SectionContent::Exercise {
                description,
                starter_code,
                solution,
            };
        }
        self
    }

    pub fn interactive_element(
        mut self,
        element_type: InteractiveElementType,
        config: serde_json::Value,
    ) -> Self {
        if let Some(ref mut section) = self.current_section {
            section.interactive_elements.push(InteractiveElement {
                element_type,
                config,
                validation_rules: Vec::new(),
            });
        }
        self
    }

    pub fn section_duration(mut self, minutes: u32) -> Self {
        if let Some(ref mut section) = self.current_section {
            section.estimated_duration = minutes;
        }
        self
    }

    pub fn completion_criteria(mut self, criteria: CompletionCriteria) -> Self {
        if let Some(ref mut section) = self.current_section {
            section.completion_criteria = criteria;
        }
        self
    }

    pub fn assessment(mut self, assessment: Assessment) -> Self {
        self.tutorial.assessment = Some(assessment);
        self
    }

    pub fn build(mut self) -> Tutorial {
        if let Some(section) = self.current_section.take() {
            self.tutorial.sections.push(section);
        }
        self.tutorial.metadata.updated_at = chrono::Utc::now();
        self.tutorial
    }
}

impl TutorialSystem {
    pub fn new(config: TutorialConfig) -> Self {
        Self {
            tutorials: Vec::new(),
            learning_paths: Vec::new(),
            progress_tracker: ProgressTracker {
                user_progress: HashMap::new(),
                global_statistics: GlobalStatistics {
                    total_users: 0,
                    tutorial_completion_rates: HashMap::new(),
                    average_scores: HashMap::new(),
                    popular_tutorials: Vec::new(),
                    common_challenges: Vec::new(),
                },
            },
            assessment_engine: AssessmentEngine {
                assessments: HashMap::new(),
                question_bank: Vec::new(),
                scoring_algorithms: HashMap::new(),
            },
            config,
        }
    }

    pub fn add_tutorial(&mut self, tutorial: Tutorial) {
        self.tutorials.push(tutorial);
    }

    pub fn create_learning_path(&mut self, path: LearningPath) {
        self.learning_paths.push(path);
    }

    pub fn generate_tutorial_from_trait(&self, trait_info: &TraitInfo) -> Tutorial {
        TutorialBuilder::new(
            format!("trait_{}", trait_info.name),
            format!("Understanding the {} Trait", trait_info.name)
        )
        .description(format!("Learn how to use and implement the {} trait in your Rust code", trait_info.name))
        .difficulty(DifficultyLevel::Intermediate)
        .duration(30)
        .learning_objective(format!("Understand the purpose and usage of {}", trait_info.name))
        .learning_objective("Implement the trait in custom types".to_string())
        .learning_objective("Use trait methods effectively".to_string())
        .category(TutorialCategory::CoreConcepts)
        .tag("traits".to_string())
        .tag("rust".to_string())
        .section("introduction".to_string(), "Introduction".to_string())
        .text_content(
            format!("The {} trait is a fundamental concept in Rust that allows you to define shared behavior across different types.\n\n{}",
                trait_info.name, trait_info.description),
            ContentFormat::Markdown
        )
        .section_duration(5)
        .section("implementation".to_string(), "Implementation Guide".to_string())
        .code_content(
            self.generate_trait_implementation_example(trait_info),
            "rust".to_string(),
            true
        )
        .interactive_element(
            InteractiveElementType::CodeEditor,
            serde_json::json!({
                "template": self.generate_trait_template(trait_info),
                "validation": "compilation"
            })
        )
        .section_duration(15)
        .section("examples".to_string(), "Practical Examples".to_string())
        .text_content(
            "Let's explore some real-world examples of using this trait:".to_string(),
            ContentFormat::Markdown
        )
        .section_duration(10)
        .build()
    }

    fn generate_trait_implementation_example(&self, trait_info: &TraitInfo) -> String {
        format!(
            "// Example implementation of {}\n\
             struct MyStruct {{\n    \
                 // Your fields here\n\
             }}\n\n\
             impl {} for MyStruct {{\n    \
                 // Implement required methods\n\
             }}",
            trait_info.name, trait_info.name
        )
    }

    fn generate_trait_template(&self, trait_info: &TraitInfo) -> String {
        format!(
            "// TODO: Implement {} for your custom type\n\
             struct YourType {{\n    \
                 // Add your fields\n\
             }}\n\n\
             impl {} for YourType {{\n    \
                 // Implement the required methods\n\
             }}",
            trait_info.name, trait_info.name
        )
    }

    pub fn generate_tutorial_from_type(&self, type_info: &TypeInfo) -> Tutorial {
        TutorialBuilder::new(
            format!("type_{}", type_info.name),
            format!("Working with {} Type", type_info.name),
        )
        .description(format!(
            "Learn how to use the {} type effectively",
            type_info.name
        ))
        .difficulty(DifficultyLevel::Beginner)
        .duration(20)
        .learning_objective(format!("Understand the {} type", type_info.name))
        .learning_objective("Create and manipulate instances".to_string())
        .category(TutorialCategory::CoreConcepts)
        .tag("types".to_string())
        .tag("rust".to_string())
        .section("overview".to_string(), "Type Overview".to_string())
        .text_content(
            format!(
                "{}\n\n**Type signature**: `{:?}`",
                type_info.description, type_info.kind
            ),
            ContentFormat::Markdown,
        )
        .section_duration(10)
        .section("usage".to_string(), "Usage Examples".to_string())
        .code_content(
            self.generate_type_usage_example(type_info),
            "rust".to_string(),
            true,
        )
        .section_duration(10)
        .build()
    }

    fn generate_type_usage_example(&self, type_info: &TypeInfo) -> String {
        format!(
            "// Example usage of {}\n\
             let instance = {}::new();\n\
             // Use the instance...",
            type_info.name, type_info.name
        )
    }

    pub fn get_recommended_tutorials(&self, user_id: &str) -> Vec<&Tutorial> {
        if let Some(progress) = self.progress_tracker.user_progress.get(user_id) {
            self.tutorials
                .iter()
                .filter(|tutorial| {
                    !progress.completed_tutorials.contains(&tutorial.id)
                        && self.meets_prerequisites(tutorial, progress)
                })
                .collect()
        } else {
            self.tutorials
                .iter()
                .filter(|tutorial| tutorial.prerequisites.is_empty())
                .collect()
        }
    }

    fn meets_prerequisites(&self, tutorial: &Tutorial, progress: &UserProgress) -> bool {
        tutorial
            .prerequisites
            .iter()
            .all(|prereq| progress.completed_tutorials.contains(prereq))
    }

    pub fn start_tutorial(&mut self, user_id: String, tutorial_id: String) -> Result<(), String> {
        if !self.tutorials.iter().any(|t| t.id == tutorial_id) {
            return Err("Tutorial not found".to_string());
        }

        let user_id_clone = user_id.clone();
        let progress = self
            .progress_tracker
            .user_progress
            .entry(user_id)
            .or_insert_with(|| UserProgress {
                user_id: user_id_clone,
                completed_tutorials: HashSet::new(),
                current_tutorial: None,
                current_section: None,
                completion_percentage: 0.0,
                time_spent: 0,
                assessment_scores: HashMap::new(),
                achievements: Vec::new(),
                learning_preferences: LearningPreferences {
                    preferred_difficulty: DifficultyLevel::Beginner,
                    learning_pace: LearningPace::Normal,
                    content_types: vec![ContentType::Text, ContentType::Code],
                    reminder_frequency: ReminderFrequency::Weekly,
                },
            });

        progress.current_tutorial = Some(tutorial_id);
        progress.current_section = None;
        progress.completion_percentage = 0.0;

        Ok(())
    }

    pub fn complete_section(
        &mut self,
        user_id: &str,
        tutorial_id: &str,
        _section_id: &str,
    ) -> Result<(), String> {
        let progress = self
            .progress_tracker
            .user_progress
            .get_mut(user_id)
            .ok_or("User not found")?;

        if progress.current_tutorial.as_ref() != Some(&tutorial_id.to_string()) {
            return Err("Tutorial not in progress".to_string());
        }

        let tutorial_info = self
            .tutorials
            .iter()
            .find(|t| t.id == tutorial_id)
            .map(|t| (t.sections.len(), t.title.clone(), t.difficulty.clone()));

        if let Some((total_sections, title, difficulty)) = tutorial_info {
            let completed_sections = progress.completed_tutorials.len();
            progress.completion_percentage =
                (completed_sections as f64 / total_sections as f64) * 100.0;

            if progress.completion_percentage >= 100.0 {
                progress.completed_tutorials.insert(tutorial_id.to_string());
                progress.current_tutorial = None;
                progress.current_section = None;

                self.award_completion_achievement_info(user_id, tutorial_id, &title, &difficulty);
            }
        }

        Ok(())
    }

    fn award_completion_achievement_info(
        &mut self,
        user_id: &str,
        tutorial_id: &str,
        title: &str,
        difficulty: &DifficultyLevel,
    ) {
        if let Some(progress) = self.progress_tracker.user_progress.get_mut(user_id) {
            let achievement = Achievement {
                id: format!("completed_{}", tutorial_id),
                title: format!("Completed: {}", title),
                description: format!("Successfully completed the {} tutorial", title),
                icon: "🎓".to_string(),
                earned_at: chrono::Utc::now(),
                rarity: match difficulty {
                    DifficultyLevel::Beginner => AchievementRarity::Common,
                    DifficultyLevel::Intermediate => AchievementRarity::Uncommon,
                    DifficultyLevel::Advanced => AchievementRarity::Rare,
                    DifficultyLevel::Expert => AchievementRarity::Epic,
                },
            };
            progress.achievements.push(achievement);
        }
    }

    pub fn get_user_progress(&self, user_id: &str) -> Option<&UserProgress> {
        self.progress_tracker.user_progress.get(user_id)
    }

    pub fn update_global_statistics(&mut self) {
        self.progress_tracker.global_statistics.total_users =
            self.progress_tracker.user_progress.len();

        for tutorial in &self.tutorials {
            let completion_count = self
                .progress_tracker
                .user_progress
                .values()
                .filter(|progress| progress.completed_tutorials.contains(&tutorial.id))
                .count();

            let completion_rate = if self.progress_tracker.global_statistics.total_users > 0 {
                completion_count as f64 / self.progress_tracker.global_statistics.total_users as f64
            } else {
                0.0
            };

            self.progress_tracker
                .global_statistics
                .tutorial_completion_rates
                .insert(tutorial.id.clone(), completion_rate);
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tutorial_builder() {
        let tutorial = TutorialBuilder::new("test".to_string(), "Test Tutorial".to_string())
            .description("Test description".to_string())
            .difficulty(DifficultyLevel::Intermediate)
            .duration(60)
            .author("Test Author".to_string())
            .build();

        assert_eq!(tutorial.id, "test");
        assert_eq!(tutorial.title, "Test Tutorial");
        assert_eq!(tutorial.description, "Test description");
        assert_eq!(tutorial.duration_minutes, 60);
        assert!(matches!(tutorial.difficulty, DifficultyLevel::Intermediate));
    }

    #[test]
    fn test_tutorial_system_creation() {
        let config = TutorialConfig::default();
        let system = TutorialSystem::new(config);
        assert_eq!(system.tutorials.len(), 0);
        assert_eq!(system.learning_paths.len(), 0);
    }

    #[test]
    fn test_user_progress_tracking() {
        let mut system = TutorialSystem::new(TutorialConfig::default());
        let tutorial = TutorialBuilder::new("test".to_string(), "Test".to_string()).build();
        system.add_tutorial(tutorial);

        let result = system.start_tutorial("user1".to_string(), "test".to_string());
        assert!(result.is_ok());

        let progress = system.get_user_progress("user1");
        assert!(progress.is_some());
        assert_eq!(progress.unwrap().current_tutorial, Some("test".to_string()));
    }

    #[test]
    fn test_section_completion() {
        let mut system = TutorialSystem::new(TutorialConfig::default());
        let tutorial = TutorialBuilder::new("test".to_string(), "Test".to_string())
            .section("section1".to_string(), "Section 1".to_string())
            .build();
        system.add_tutorial(tutorial);

        system
            .start_tutorial("user1".to_string(), "test".to_string())
            .unwrap();
        let result = system.complete_section("user1", "test", "section1");
        assert!(result.is_ok());
    }

    #[test]
    fn test_tutorial_generation_from_trait() {
        let system = TutorialSystem::new(TutorialConfig::default());
        let trait_info = TraitInfo {
            name: "Display".to_string(),
            path: "std::fmt::Display".to_string(),
            description: "Trait for formatting output".to_string(),
            methods: vec![],
            associated_types: vec![],
            generics: vec![],
            supertraits: vec![],
            implementations: vec![],
        };

        let tutorial = system.generate_tutorial_from_trait(&trait_info);
        assert_eq!(tutorial.id, "trait_Display");
        assert!(tutorial.title.contains("Display"));
        assert!(!tutorial.sections.is_empty());
    }
}
