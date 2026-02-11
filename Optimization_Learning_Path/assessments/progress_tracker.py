"""
Progress Tracking System for Optimization Learning Path
Tracks video completion, challenge progress, and learning outcomes
"""

import json
import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class VideoProgress:
    """Track progress for a single video."""
    video_id: str
    title: str
    url: str
    watched: bool = False
    watch_date: Optional[str] = None
    notes: str = ""
    rating: Optional[int] = None  # 1-5 scale

@dataclass
class ChallengeProgress:
    """Track progress for a coding challenge."""
    challenge_id: str
    title: str
    file_path: str
    completed: bool = False
    completion_date: Optional[str] = None
    code_quality: Optional[int] = None  # 1-5 scale
    time_spent: Optional[float] = None  # hours
    notes: str = ""
    test_results: Optional[Dict[str, Any]] = None

@dataclass
class QuizResult:
    """Track quiz and assessment results."""
    quiz_id: str
    title: str
    score: float  # percentage
    max_score: float
    attempt_date: str
    time_taken: float  # minutes
    answers: Dict[str, Any]

@dataclass
class LearningPath:
    """Track overall progress through a learning path."""
    path_name: str
    total_videos: int
    total_challenges: int
    total_quizzes: int
    videos_completed: int = 0
    challenges_completed: int = 0
    quizzes_completed: int = 0
    start_date: Optional[str] = None
    target_completion_date: Optional[str] = None
    current_week: int = 1

class ProgressTracker:
    """
    Comprehensive progress tracking system for the optimization learning path.
    """
    
    def __init__(self, data_file: str = "learning_progress.json"):
        """
        Initialize the progress tracker.
        
        Args:
            data_file: Path to the JSON file storing progress data
        """
        self.data_file = Path(data_file)
        self.data = self._load_data()
        
    def _load_data(self) -> Dict[str, Any]:
        """Load progress data from JSON file."""
        if self.data_file.exists():
            with open(self.data_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'convex_optimization': {
                    'videos': {},
                    'challenges': {},
                    'quizzes': {},
                    'path_info': {
                        'path_name': 'Convex Optimization',
                        'total_videos': 9,
                        'total_challenges': 9,
                        'total_quizzes': 4,
                        'videos_completed': 0,
                        'challenges_completed': 0,
                        'quizzes_completed': 0,
                        'start_date': None,
                        'target_completion_date': None,
                        'current_week': 1
                    }
                },
                'dynamic_programming': {
                    'videos': {},
                    'challenges': {},
                    'quizzes': {},
                    'path_info': {
                        'path_name': 'Dynamic Programming',
                        'total_videos': 9,
                        'total_challenges': 9,
                        'total_quizzes': 4,
                        'videos_completed': 0,
                        'challenges_completed': 0,
                        'quizzes_completed': 0,
                        'start_date': None,
                        'target_completion_date': None,
                        'current_week': 1
                    }
                }
            }
    
    def _save_data(self):
        """Save progress data to JSON file."""
        with open(self.data_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def mark_video_watched(self, path_name: str, video_id: str, 
                          rating: Optional[int] = None, notes: str = ""):
        """
        Mark a video as watched.
        
        Args:
            path_name: Name of the learning path
            video_id: Unique identifier for the video
            rating: Optional rating (1-5)
            notes: Optional notes about the video
        """
        if path_name not in self.data:
            self.data[path_name] = {'videos': {}, 'challenges': {}, 'quizzes': {}, 'path_info': {}}
        
        if 'videos' not in self.data[path_name]:
            self.data[path_name]['videos'] = {}
        
        video_data = {
            'video_id': video_id,
            'title': f"Lecture {video_id}",
            'url': f"https://youtube.com/watch?v=video_{video_id}",
            'watched': True,
            'watch_date': datetime.datetime.now().isoformat(),
            'notes': notes,
            'rating': rating
        }
        
        self.data[path_name]['videos'][video_id] = video_data
        
        # Update path completion count
        if 'path_info' in self.data[path_name]:
            self.data[path_name]['path_info']['videos_completed'] = len([
                v for v in self.data[path_name]['videos'].values() if v.get('watched', False)
            ])
        
        self._save_data()
    
    def mark_challenge_completed(self, path_name: str, challenge_id: str,
                               code_quality: Optional[int] = None, 
                               time_spent: Optional[float] = None,
                               notes: str = "", test_results: Optional[Dict] = None):
        """
        Mark a coding challenge as completed.
        
        Args:
            path_name: Name of the learning path
            challenge_id: Unique identifier for the challenge
            code_quality: Optional quality rating (1-5)
            time_spent: Optional time spent in hours
            notes: Optional notes about the challenge
            test_results: Optional test results
        """
        if path_name not in self.data:
            self.data[path_name] = {'videos': {}, 'challenges': {}, 'quizzes': {}, 'path_info': {}}
        
        if 'challenges' not in self.data[path_name]:
            self.data[path_name]['challenges'] = {}
        
        challenge_data = {
            'challenge_id': challenge_id,
            'title': f"Challenge {challenge_id}",
            'file_path': f"challenges/{challenge_id}.py",
            'completed': True,
            'completion_date': datetime.datetime.now().isoformat(),
            'code_quality': code_quality,
            'time_spent': time_spent,
            'notes': notes,
            'test_results': test_results
        }
        
        self.data[path_name]['challenges'][challenge_id] = challenge_data
        
        # Update path completion count
        if 'path_info' in self.data[path_name]:
            self.data[path_name]['path_info']['challenges_completed'] = len([
                c for c in self.data[path_name]['challenges'].values() if c.get('completed', False)
            ])
        
        self._save_data()
    
    def record_quiz_result(self, path_name: str, quiz_id: str, score: float,
                          max_score: float, time_taken: float, answers: Dict[str, Any]):
        """
        Record quiz results.
        
        Args:
            path_name: Name of the learning path
            quiz_id: Unique identifier for the quiz
            score: Score achieved
            max_score: Maximum possible score
            time_taken: Time taken in minutes
            answers: Dictionary of answers
        """
        if path_name not in self.data:
            self.data[path_name] = {'videos': {}, 'challenges': {}, 'quizzes': {}, 'path_info': {}}
        
        if 'quizzes' not in self.data[path_name]:
            self.data[path_name]['quizzes'] = {}
        
        quiz_data = {
            'quiz_id': quiz_id,
            'title': f"Quiz {quiz_id}",
            'score': score,
            'max_score': max_score,
            'attempt_date': datetime.datetime.now().isoformat(),
            'time_taken': time_taken,
            'answers': answers
        }
        
        self.data[path_name]['quizzes'][quiz_id] = quiz_data
        
        # Update path completion count
        if 'path_info' in self.data[path_name]:
            self.data[path_name]['path_info']['quizzes_completed'] = len([
                q for q in self.data[path_name]['quizzes'].values()
            ])
        
        self._save_data()
    
    def get_progress_summary(self, path_name: str) -> Dict[str, Any]:
        """
        Get a summary of progress for a learning path.
        
        Args:
            path_name: Name of the learning path
            
        Returns:
            Dictionary with progress summary
        """
        if path_name not in self.data:
            return {'error': f'Path {path_name} not found'}
        
        path_data = self.data[path_name]
        path_info = path_data.get('path_info', {})
        
        videos_completed = len([v for v in path_data.get('videos', {}).values() if v.get('watched', False)])
        challenges_completed = len([c for c in path_data.get('challenges', {}).values() if c.get('completed', False)])
        quizzes_completed = len(path_data.get('quizzes', {}))
        
        total_videos = path_info.get('total_videos', 0)
        total_challenges = path_info.get('total_challenges', 0)
        total_quizzes = path_info.get('total_quizzes', 0)
        
        return {
            'path_name': path_name,
            'overall_progress': {
                'videos': f"{videos_completed}/{total_videos} ({videos_completed/total_videos*100:.1f}%)" if total_videos > 0 else "0/0",
                'challenges': f"{challenges_completed}/{total_challenges} ({challenges_completed/total_challenges*100:.1f}%)" if total_challenges > 0 else "0/0",
                'quizzes': f"{quizzes_completed}/{total_quizzes} ({quizzes_completed/total_quizzes*100:.1f}%)" if total_quizzes > 0 else "0/0"
            },
            'completion_percentage': (videos_completed + challenges_completed + quizzes_completed) / (total_videos + total_challenges + total_quizzes) * 100 if (total_videos + total_challenges + total_quizzes) > 0 else 0,
            'current_week': path_info.get('current_week', 1),
            'start_date': path_info.get('start_date'),
            'target_completion_date': path_info.get('target_completion_date')
        }
    
    def get_all_progress(self) -> Dict[str, Any]:
        """Get progress summary for all learning paths."""
        return {
            path_name: self.get_progress_summary(path_name)
            for path_name in self.data.keys()
        }
    
    def generate_progress_report(self) -> str:
        """Generate a comprehensive progress report."""
        report = []
        report.append("Optimization Learning Path - Progress Report")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        all_progress = self.get_all_progress()
        
        for path_name, progress in all_progress.items():
            if 'error' in progress:
                continue
                
            report.append(f"📚 {progress['path_name']}")
            report.append("-" * 30)
            report.append(f"Overall Completion: {progress['completion_percentage']:.1f}%")
            report.append(f"Current Week: {progress['current_week']}")
            report.append("")
            
            overall = progress['overall_progress']
            report.append(f"  Videos:     {overall['videos']}")
            report.append(f"  Challenges: {overall['challenges']}")
            report.append(f"  Quizzes:    {overall['quizzes']}")
            report.append("")
        
        return "\n".join(report)
    
    def set_learning_goal(self, path_name: str, target_completion_date: str):
        """
        Set a target completion date for a learning path.
        
        Args:
            path_name: Name of the learning path
            target_completion_date: Target completion date (ISO format)
        """
        if path_name not in self.data:
            self.data[path_name] = {'videos': {}, 'challenges': {}, 'quizzes': {}, 'path_info': {}}
        
        if 'path_info' not in self.data[path_name]:
            self.data[path_name]['path_info'] = {}
        
        self.data[path_name]['path_info']['target_completion_date'] = target_completion_date
        
        if not self.data[path_name]['path_info'].get('start_date'):
            self.data[path_name]['path_info']['start_date'] = datetime.datetime.now().isoformat()
        
        self._save_data()


def main():
    """
    Demo of the progress tracking system.
    """
    print("Progress Tracking System Demo")
    print("=" * 40)
    
    # Initialize tracker
    tracker = ProgressTracker()
    
    # Set learning goals
    tracker.set_learning_goal("convex_optimization", "2024-03-01")
    tracker.set_learning_goal("dynamic_programming", "2024-03-15")
    
    # Mark some progress
    tracker.mark_video_watched("convex_optimization", "lecture_01", rating=5, notes="Great introduction!")
    tracker.mark_challenge_completed("convex_optimization", "convex_function_checker", 
                                   code_quality=4, time_spent=2.5, notes="Learned a lot about convexity testing")
    
    tracker.mark_video_watched("dynamic_programming", "lecture_01", rating=4)
    tracker.mark_challenge_completed("dynamic_programming", "fibonacci_comparison", 
                                   code_quality=5, time_spent=3.0)
    
    # Record quiz results
    tracker.record_quiz_result("convex_optimization", "quiz_week1", 85.0, 100.0, 25.0, 
                              {"q1": "correct", "q2": "correct", "q3": "incorrect"})
    
    # Generate and display report
    report = tracker.generate_progress_report()
    print(report)
    
    # Show individual progress
    print("\nDetailed Progress:")
    for path_name in ["convex_optimization", "dynamic_programming"]:
        summary = tracker.get_progress_summary(path_name)
        print(f"\n{path_name}:")
        print(f"  Completion: {summary['completion_percentage']:.1f}%")
        print(f"  Week: {summary['current_week']}")


if __name__ == "__main__":
    main()
