#!/usr/bin/env python3
"""
Resume Optimization Agent
Usage: python resume_optimizer.py <json_file_path>
Then input job description when prompted.
"""

import json
import os
import re
import sys
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

@dataclass
class OptimizationResult:
    ats_score: int
    missing_keywords: List[str]
    recommendations: List[str]
    keyword_density: Dict[str, int]
    sections_to_improve: List[str]

class ResumeOptimizer:
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ Error: OPENAI_API_KEY not found in environment variables")
            print("Create a .env file with: OPENAI_API_KEY=your_key_here")
            sys.exit(1)
        self.client = OpenAI(api_key=api_key)
        
    def load_resume_data(self, file_path: str) -> Dict:
        """Load resume data from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Error: File '{file_path}' not found")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"❌ Error: Invalid JSON format in '{file_path}'")
            sys.exit(1)
    
    def extract_keywords_from_jd(self, job_description: str) -> Set[str]:
        """Extract important keywords from job description"""
        prompt = f"""
        Extract key technical skills, tools, programming languages, frameworks, and important buzzwords from this job description.
        Return ONLY a comma-separated list of keywords (no explanations, no numbered lists).
        Focus on: skills, technologies, tools, certifications, methodologies.
        
        Job Description:
        {job_description[:1500]}
        
        Keywords:"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.1
            )
            
            keywords_text = response.choices[0].message.content.strip()
            keywords = [kw.strip().lower() for kw in keywords_text.split(',') if kw.strip()]
            return set(keywords[:20])  # Limit to top 20 keywords
            
        except Exception as e:
            print(f"⚠️  Warning: Could not extract keywords from JD: {e}")
            return set()
    
    def analyze_keyword_presence(self, resume_text: str, jd_keywords: Set[str]) -> Tuple[Dict[str, int], List[str]]:
        """Analyze which keywords are present/missing in resume"""
        resume_lower = resume_text.lower()
        keyword_density = {}
        missing_keywords = []
        
        for keyword in jd_keywords:
            # Use word boundaries for exact matches
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            matches = re.findall(pattern, resume_lower)
            count = len(matches)
            
            if count > 0:
                keyword_density[keyword] = count
            else:
                missing_keywords.append(keyword)
        
        return keyword_density, missing_keywords
    
    def calculate_ats_score(self, resume_text: str, job_description: str, 
                          keyword_density: Dict[str, int], missing_keywords: List[str]) -> int:
        """Calculate ATS score using standardized GPT-4o algorithm"""
        prompt = f"""You are an ATS (Applicant Tracking System) scoring algorithm. Calculate a standardized ATS score from 0-100 using this EXACT methodology:

ALGORITHM:
1. Keyword Match Rate = (Found Keywords / Total JD Keywords) × 60 points
2. Keyword Frequency Bonus = min(Total keyword occurrences × 1.5, 25) points  
3. Content Relevance = Assess resume relevance to JD (0-15 points)

DATA:
- Found Keywords ({len(keyword_density)}): {list(keyword_density.keys())[:10]}
- Missing Keywords ({len(missing_keywords)}): {missing_keywords[:10]}
- Keyword Frequencies: {dict(list(keyword_density.items())[:8])}
- Total JD Keywords: {len(keyword_density) + len(missing_keywords)}

CALCULATE:
Step 1: Keyword Match = ({len(keyword_density)}/{len(keyword_density) + len(missing_keywords)}) × 60 = ?
Step 2: Frequency Bonus = min({sum(keyword_density.values())} × 1.5, 25) = ?
Step 3: Content Relevance (0-15) based on resume-JD alignment = ?

Final Score = Step1 + Step2 + Step3

Respond with ONLY the final numeric score (0-100)."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )
            
            # Extract numeric score from response
            score_text = response.choices[0].message.content.strip()
            score_match = re.search(r'\b(\d{1,3})\b', score_text)
            
            if score_match:
                score = int(score_match.group(1))
                return min(max(score, 0), 100)
            else:
                # Fallback calculation if GPT doesn't return valid score
                return self._fallback_score_calculation(keyword_density, missing_keywords)
                
        except Exception as e:
            print(f"⚠️  Warning: GPT scoring failed ({e}), using fallback calculation")
            return self._fallback_score_calculation(keyword_density, missing_keywords)
    
    def _fallback_score_calculation(self, keyword_density: Dict[str, int], missing_keywords: List[str]) -> int:
        """Fallback ATS score calculation if GPT fails"""
        total_keywords = len(keyword_density) + len(missing_keywords)
        if total_keywords == 0:
            return 50
        
        # Use same algorithm as defined in prompt
        keyword_match = (len(keyword_density) / total_keywords) * 60
        frequency_bonus = min(sum(keyword_density.values()) * 1.5, 25)
        content_relevance = 10  # Default middle score
        
        score = int(keyword_match + frequency_bonus + content_relevance)
        return min(max(score, 0), 100)
    
    def generate_recommendations(self, missing_keywords: List[str]) -> List[str]:
        """Generate concise recommendations"""
        recommendations = []
        
        if len(missing_keywords) > 0:
            top_missing = missing_keywords[:6]
            recommendations.append(f"Add: {', '.join(top_missing)}")
        
        if len(missing_keywords) > 8:
            recommendations.append("Align experience with JD requirements")
        
        if len(missing_keywords) > 5:
            recommendations.append("Expand technical skills section")
            
        if len(missing_keywords) > 12:
            recommendations.append("Consider rewriting project descriptions")
        
        return recommendations[:4]  # Max 4 recommendations
    
    def identify_sections_to_improve(self, missing_keywords: List[str]) -> List[str]:
        """Identify which resume sections need improvement"""
        sections = set()
        
        # Technical skills indicators
        tech_keywords = ['python', 'java', 'react', 'sql', 'aws', 'docker', 'kubernetes', 
                        'javascript', 'node', 'angular', 'vue', 'mongodb', 'postgresql']
        if any(kw in missing_keywords for kw in tech_keywords):
            sections.add("Skills")
        
        # Experience indicators
        exp_keywords = ['management', 'leadership', 'agile', 'scrum', 'team', 'project']
        if any(kw in missing_keywords for kw in exp_keywords):
            sections.add("Experience")
            
        # Education/Certification indicators
        edu_keywords = ['certification', 'degree', 'course', 'training']
        if any(kw in missing_keywords for kw in edu_keywords):
            sections.add("Education")
        
        return list(sections)[:3]
    
    def optimize_resume(self, resume_file_path: str, job_description: str) -> OptimizationResult:
        """Main optimization function"""
        print("🔍 Loading resume data...")
        resume_data = self.load_resume_data(resume_file_path)
        resume_text = resume_data.get('cleaned_text', resume_data.get('raw_text', ''))
        
        if not resume_text:
            print("❌ Error: No resume text found in JSON file")
            sys.exit(1)
        
        print("🤖 Extracting keywords from job description...")
        jd_keywords = self.extract_keywords_from_jd(job_description)
        
        if not jd_keywords:
            print("⚠️  Warning: No keywords extracted from job description")
        
        print("📊 Analyzing keyword presence...")
        keyword_density, missing_keywords = self.analyze_keyword_presence(resume_text, jd_keywords)
        
        print("📈 Calculating ATS score...")
        ats_score = self.calculate_ats_score(resume_text, job_description, keyword_density, missing_keywords)
        
        print("💡 Generating recommendations...")
        recommendations = self.generate_recommendations(missing_keywords)
        sections_to_improve = self.identify_sections_to_improve(missing_keywords)
        
        return OptimizationResult(
            ats_score=ats_score,
            missing_keywords=missing_keywords[:10],  # Top 10 missing
            recommendations=recommendations,
            keyword_density=dict(list(keyword_density.items())[:10]),  # Top 10 found
            sections_to_improve=sections_to_improve
        )
    
    def save_results(self, result: OptimizationResult, output_file: str):
        """Save optimization result to JSON file"""
        output_data = {
            "ats": result.ats_score,
            "missing": result.missing_keywords,
            "recs": result.recommendations,
            "density": result.keyword_density,
            "improve": result.sections_to_improve
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, separators=(',', ':'))
        
        return output_data

def print_results(result: OptimizationResult):
    """Print formatted results"""
    print("\n" + "="*50)
    print("📋 RESUME OPTIMIZATION RESULTS")
    print("="*50)
    print(f"🎯 ATS Score: {result.ats_score}/100")
    
    if result.ats_score >= 80:
        print("   ✅ Excellent match!")
    elif result.ats_score >= 60:
        print("   ⚠️  Good match, room for improvement")
    else:
        print("   ❌ Needs significant optimization")
    
    print(f"\n📈 Keywords Found: {len(result.keyword_density)}")
    if result.keyword_density:
        top_keywords = dict(list(result.keyword_density.items())[:5])
        for kw, count in top_keywords.items():
            print(f"   • {kw}: {count}")
    
    print(f"\n❌ Missing Keywords ({len(result.missing_keywords)}):")
    for kw in result.missing_keywords[:8]:
        print(f"   • {kw}")
    
    print(f"\n💡 Recommendations:")
    for rec in result.recommendations:
        print(f"   • {rec}")
    
    if result.sections_to_improve:
        print(f"\n🔧 Sections to Improve:")
        for section in result.sections_to_improve:
            print(f"   • {section}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python resume_optimizer.py <json_file_path>")
        print("Example: python resume_optimizer.py paste.txt")
        sys.exit(1)
    
    json_file_path = sys.argv[1]
    
    print("🚀 Resume Optimization Agent")
    print("="*30)
    
    # Get job description from user
    print("\n📝 Please paste the job description below:")
    print("(Press Ctrl+D on Unix/Linux/Mac or Ctrl+Z on Windows when done)")
    
    try:
        job_description = sys.stdin.read().strip()
    except KeyboardInterrupt:
        print("\n👋 Optimization cancelled.")
        sys.exit(0)
    
    if not job_description:
        print("❌ Error: No job description provided")
        sys.exit(1)
    
    # Initialize optimizer and run analysis
    optimizer = ResumeOptimizer()
    
    try:
        result = optimizer.optimize_resume(json_file_path, job_description)
        
        # Print results
        print_results(result)
        
        # Save to file
        output_file = 'optimization_result.json'
        optimizer.save_results(result, output_file)
        print(f"\n💾 Results saved to: {output_file}")
        
    except KeyboardInterrupt:
        print("\n👋 Optimization cancelled.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during optimization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()