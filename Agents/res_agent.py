import json
import os
import re
from openai import OpenAI
from typing import Dict, List, Any

class ResumeOptimizerAgent:
    def __init__(self):
        """Initialize the Resume Optimizer Agent with OpenAI client"""
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def extract_resume_data(self, resume_json: Dict[str, Any]) -> Dict[str, str]:
        """Extract structured data from resume JSON using GPT-4"""
        
        raw_text = resume_json.get('raw_text', '')
        
        extraction_prompt = """
You are a resume data extraction expert. Extract the following information from the resume text and return it as a JSON object with these exact keys:

{
  "candidate_name": "Full name",
  "email": "Email address", 
  "phone": "Phone number",
  "location": "City, State, Country",
  "linkedin_url": "LinkedIn profile URL",
  "github_url": "GitHub profile URL",
  "university_name": "Current university/college name",
  "university_location": "University city",
  "degree_name": "Degree type (e.g., Bachelor of Engineering)",
  "major": "Field of study/major",
  "university_start": "Start date",
  "university_end": "End date", 
  "gpa": "GPA or percentage",
  "previous_college": "Previous college name",
  "previous_college_location": "Previous college location",
  "previous_degree": "Previous degree type",
  "previous_stream": "Previous academic stream",
  "previous_percentage": "Previous percentage/grade",
  "previous_start": "Previous education start",
  "previous_end": "Previous education end",
  "summary_points": ["List of 3 summary points"],
  "projects": [
    {
      "name": "Project name",
      "technologies": "Technologies used",
      "description": "Project description",
      "link": "Project link if available"
    }
  ],
  "cp_achievements": ["List of competitive programming achievements"],
  "hackathons": ["List of hackathons and involvements"],
  "programming_languages": "Comma-separated programming languages",
  "core_skills": "Comma-separated core technical skills",
  "tools": "Comma-separated tools and frameworks",
  "additional_skills": "Any additional skill categories and their values",
  "spoken_languages": "Comma-separated spoken languages",
  "interests": "Comma-separated hobbies and interests"
}

Extract only the information that's clearly present in the resume. Use "Not provided" for missing information.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": extraction_prompt},
                    {"role": "user", "content": f"Resume text:\n{raw_text}"}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            extracted_data = json.loads(response.choices[0].message.content.strip())
            return extracted_data
            
        except Exception as e:
            print(f"Error extracting resume data: {str(e)}")
            return self._get_default_template_data()
    
    def _get_default_template_data(self) -> Dict[str, str]:
        """Return default template data structure"""
        return {
            "candidate_name": "[CANDIDATE_NAME]",
            "email": "[EMAIL]",
            "phone": "[PHONE]",
            "location": "[LOCATION]",
            "linkedin_url": "[LINKEDIN_URL]",
            "github_url": "[GITHUB_URL]",
            "university_name": "[UNIVERSITY]",
            "university_location": "[UNI_LOCATION]",
            "degree_name": "[DEGREE]",
            "major": "[MAJOR]",
            "university_start": "[UNI_START]",
            "university_end": "[UNI_END]",
            "gpa": "[GPA]",
            "previous_college": "[PREV_COLLEGE]",
            "previous_college_location": "[PREV_LOCATION]",
            "previous_degree": "[PREV_DEGREE]",
            "previous_stream": "[PREV_STREAM]",
            "previous_percentage": "[PREV_PERCENTAGE]",
            "previous_start": "[PREV_START]",
            "previous_end": "[PREV_END]",
            "summary_points": ["[SUMMARY_POINT_1]", "[SUMMARY_POINT_2]", "[SUMMARY_POINT_3]"],
            "projects": [],
            "cp_achievements": ["[CP_ACHIEVEMENT_1]"],
            "hackathons": ["[HACKATHON_1]"],
            "programming_languages": "[LANGUAGES]",
            "core_skills": "[CORE_SKILLS]",
            "tools": "[TOOLS]",
            "additional_skills": "[ADDITIONAL_SKILLS]",
            "spoken_languages": "[SPOKEN_LANGUAGES]",
            "interests": "[INTERESTS]"
        }
    
    def fill_template(self, template_latex: str, extracted_data: Dict[str, Any], 
                     missing_keywords: List[str] = None) -> str:
        """Fill the LaTeX template with extracted data and optimize with missing keywords"""
        
        # Create a copy to work with
        filled_latex = template_latex
        
        # Basic replacements
        replacements = {
            '[CANDIDATE_NAME]': extracted_data.get('candidate_name', 'John Doe'),
            '[CITY]': extracted_data.get('location', 'City').split(',')[0] if ',' in extracted_data.get('location', '') else extracted_data.get('location', 'City'),
            '[STATE]': extracted_data.get('location', 'State, Country').split(',')[1].strip() if ',' in extracted_data.get('location', '') else 'State',
            '[COUNTRY]': extracted_data.get('location', 'City, State, Country').split(',')[-1].strip() if ',' in extracted_data.get('location', '') else 'Country',
            '[PHONE_NUMBER]': extracted_data.get('phone', '1234567890'),
            '[EMAIL]': extracted_data.get('email', 'email@example.com'),
            '[LINKEDIN_URL]': extracted_data.get('linkedin_url', 'https://linkedin.com/in/profile'),
            '[GITHUB_URL]': extracted_data.get('github_url', 'https://github.com/username'),
            '[UNIVERSITY_NAME]': extracted_data.get('university_name', 'University Name'),
            '[UNIVERSITY_LOCATION]': extracted_data.get('university_location', 'Location'),
            '[START_DATE]': extracted_data.get('university_start', 'Start Date'),
            '[END_DATE]': extracted_data.get('university_end', 'End Date'),
            '[DEGREE_NAME]': extracted_data.get('degree_name', 'Bachelor of Science'),
            '[MAJOR]': extracted_data.get('major', 'Computer Science'),
            '[GPA_OR_PERCENTAGE]': extracted_data.get('gpa', 'GPA: X.X/10'),
            '[COLLEGE_NAME]': extracted_data.get('previous_college', 'Previous College'),
            '[COLLEGE_LOCATION]': extracted_data.get('previous_college_location', 'Location'),
            '[DEGREE_TYPE]': extracted_data.get('previous_degree', 'Degree Type'),
            '[STREAM]': extracted_data.get('previous_stream', 'Stream'),
            '[PERCENTAGE]': extracted_data.get('previous_percentage', 'XX.X%'),
            '[PROGRAMMING_LANGUAGES]': extracted_data.get('programming_languages', 'Python, Java, C++'),
            '[CORE_TECHNICAL_SKILLS]': extracted_data.get('core_skills', 'Data Structures, Algorithms'),
            '[TOOLS_AND_FRAMEWORKS]': extracted_data.get('tools', 'Git, VS Code'),
            '[ADDITIONAL_SKILL_CATEGORY]': 'Additional Skills',
            '[ADDITIONAL_SKILLS]': extracted_data.get('additional_skills', 'Various Skills'),
            '[SPOKEN_LANGUAGES]': extracted_data.get('spoken_languages', 'English (Fluent)'),
            '[HOBBIES_AND_INTERESTS]': extracted_data.get('interests', 'Reading, Programming')
        }
        
        # Apply basic replacements
        for placeholder, value in replacements.items():
            filled_latex = filled_latex.replace(placeholder, value)
        
        # Handle summary points
        summary_points = extracted_data.get('summary_points', ['Summary point 1', 'Summary point 2', 'Summary point 3'])
        for i, point in enumerate(summary_points[:3], 1):
            filled_latex = filled_latex.replace(f'[SUMMARY_POINT_{i}]', point)
        
        # Handle projects
        projects = extracted_data.get('projects', [])
        project_sections = []
        for i, project in enumerate(projects[:4], 1):
            project_text = f"\\textbf{{{project.get('name', f'Project {i}')}}} \\textit{{{project.get('technologies', 'Technologies')}}} \\\\\n{project.get('description', 'Project description')} \\href{{{project.get('link', '#')}}}{{(link)}} \\\\[6pt]"
            project_sections.append(project_text)
            
        # Replace project placeholders
        for i in range(1, 5):
            if i <= len(project_sections):
                filled_latex = filled_latex.replace(f'\\textbf{{[PROJECT_{i}_NAME]}} \\textit{{[TECHNOLOGIES_USED]}} \\\\\n[PROJECT_{i}_DESCRIPTION] \\href{{[PROJECT_{i}_LINK]}}{{(link)}} \\\\[6pt]', project_sections[i-1])
            else:
                # Remove unused project sections
                filled_latex = re.sub(f'\\\\textbf{{\\[PROJECT_{i}_NAME\\]}}.*?\\\\\\[6pt\\]', '', filled_latex, flags=re.DOTALL)
        
        # Handle competitive programming achievements
        cp_achievements = extracted_data.get('cp_achievements', ['CP achievement 1', 'CP achievement 2', 'CP achievement 3'])
        for i, achievement in enumerate(cp_achievements[:3], 1):
            filled_latex = filled_latex.replace(f'[CP_ACHIEVEMENT_{i}]', achievement)
        
        # Handle hackathons and involvements
        hackathons = extracted_data.get('hackathons', ['Hackathon 1', 'Hackathon 2', 'Hackathon 3', 'Involvement 1'])
        for i, hackathon in enumerate(hackathons[:4], 1):
            filled_latex = filled_latex.replace(f'[HACKATHON_OR_INVOLVEMENT_{i}]', hackathon)
        
        return filled_latex
        
    def optimize_resume(self, ats_analysis: Dict[str, Any], original_latex: str, 
                       resume_json: Dict[str, Any], job_description: str = "") -> str:
        """
        Optimize the LaTeX resume based on ATS analysis
        
        Args:
            ats_analysis: Dictionary containing ATS score, missing keywords, recommendations
            original_latex: Original LaTeX code of the resume (can be template or filled)
            resume_json: Parsed resume data in JSON format
            job_description: Optional job description for context
            
        Returns:
            Optimized LaTeX code
        """
        
        # Check if it's a template
        is_template = '[CANDIDATE_NAME]' in original_latex
        
        if is_template:
            print("📋 Detected template format - extracting data and filling template...")
            
            # Extract structured data from resume JSON
            extracted_data = self.extract_resume_data(resume_json)
            
            # Fill the template with extracted data
            filled_latex = self.fill_template(original_latex, extracted_data, 
                                            ats_analysis.get('missing', []))
            
            # Now optimize the filled template
            optimized_latex = self._optimize_filled_resume(filled_latex, ats_analysis, 
                                                         resume_json, job_description)
        else:
            print("📄 Processing existing resume - optimizing content...")
            # Direct optimization of existing resume
            optimized_latex = self._optimize_filled_resume(original_latex, ats_analysis, 
                                                         resume_json, job_description)
        
        return optimized_latex
    
    def _optimize_filled_resume(self, filled_latex: str, ats_analysis: Dict[str, Any], 
                              resume_json: Dict[str, Any], job_description: str) -> str:
        """Optimize an already filled resume with ATS improvements"""
        
        # Extract key information from ATS analysis
        missing_keywords = ats_analysis.get('missing', [])
        recommendations = ats_analysis.get('recs', [])
        keyword_density = ats_analysis.get('density', {})
        areas_to_improve = ats_analysis.get('improve', [])
        current_ats_score = ats_analysis.get('ats', 0)
        
        # Build optimization prompt
        optimization_prompt = self._build_optimization_prompt(
            missing_keywords, recommendations, keyword_density, 
            areas_to_improve, current_ats_score, job_description
        )
        
        # Generate optimized LaTeX using GPT-4
        optimized_latex = self._generate_optimized_latex_content(
            optimization_prompt, filled_latex, resume_json
        )
        
        return optimized_latex
    
    def _build_optimization_prompt(self, missing_keywords: List[str], 
                                 recommendations: List[str], keyword_density: Dict[str, int], 
                                 areas_to_improve: List[str], ats_score: int, 
                                 job_description: str) -> str:
        """Build the optimization prompt for GPT-4"""
        
        prompt = f"""
You are an expert resume optimizer. Your task is to modify a LaTeX resume to improve its ATS compatibility and alignment with job requirements.

CURRENT ATS SCORE: {ats_score}/100

MISSING KEYWORDS TO INCORPORATE:
{', '.join(missing_keywords)}

SPECIFIC RECOMMENDATIONS:
{chr(10).join(f'• {rec}' for rec in recommendations)}

AREAS THAT NEED IMPROVEMENT:
{', '.join(areas_to_improve)}

CURRENT KEYWORD DENSITY:
{json.dumps(keyword_density, indent=2)}

{f'JOB DESCRIPTION CONTEXT: {job_description}' if job_description else ''}

OPTIMIZATION INSTRUCTIONS:
1. **Strategically integrate missing keywords** naturally into existing sections
2. **Enhance project descriptions** to highlight relevant technical skills
3. **Optimize the Skills section** by reorganizing and adding missing technologies
4. **Improve keyword density** for underrepresented but important terms
5. **Maintain professional formatting** and LaTeX structure
6. **Ensure natural language flow** - avoid keyword stuffing
7. **Quantify achievements** where possible with metrics and percentages
8. **Align project descriptions** with job requirements
9. **Add relevant technical details** to demonstrate expertise
10. **Keep the same overall structure** but enhance content strategically

SPECIFIC ENHANCEMENT STRATEGIES:
- For Skills section: Group related technologies, add missing keywords naturally
- For Projects: Rewrite descriptions to emphasize relevant technologies and outcomes
- For Summary: Incorporate key missing keywords that align with the candidate's background
- For Experience descriptions: Add technical depth and relevant methodologies

Remember to:
- Maintain the exact LaTeX formatting and structure
- Keep all existing personal information unchanged
- Ensure all modifications are truthful and based on existing experience
- Use action verbs and technical terminology appropriately
- Balance keyword optimization with readability
"""
        return prompt
    
    def _generate_optimized_latex_content(self, optimization_prompt: str, 
                                        filled_latex: str, resume_json: Dict[str, Any]) -> str:
        """Generate optimized LaTeX content using GPT-4"""
        
        messages = [
            {
                "role": "system", 
                "content": optimization_prompt
            },
            {
                "role": "user", 
                "content": f"""
Here is the LaTeX resume that needs keyword optimization:

```latex
{filled_latex}
```

Please enhance this resume by strategically incorporating the missing keywords while maintaining natural language flow and professional quality. Return ONLY the optimized LaTeX code.
"""
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.3,
                max_tokens=4000,
            )
            
            optimized_latex = response.choices[0].message.content.strip()
            optimized_latex = self._clean_latex_output(optimized_latex)
            
            return optimized_latex
            
        except Exception as e:
            print(f"Error generating optimized resume: {str(e)}")
            return filled_latex
    
    def _clean_latex_output(self, latex_content: str) -> str:
        """Clean up the LaTeX output from GPT-4"""
        
        # Remove markdown code blocks if present
        latex_content = re.sub(r'^```latex\n', '', latex_content)
        latex_content = re.sub(r'^```\n', '', latex_content)
        latex_content = re.sub(r'\n```$', '', latex_content)
        latex_content = re.sub(r'^```$', '', latex_content, flags=re.MULTILINE)
        
        # Ensure proper LaTeX document structure
        if not latex_content.strip().startswith('\\documentclass'):
            # If it doesn't start with documentclass, assume it's just the content
            # and we need to extract it properly
            pass
        
        return latex_content.strip()
    
    def save_optimized_resume(self, optimized_latex: str, output_path: str) -> bool:
        """Save the optimized LaTeX resume to a file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(optimized_latex)
            print(f"Optimized resume saved to: {output_path}")
            return True
        except Exception as e:
            print(f"Error saving optimized resume: {str(e)}")
            return False
    
    def get_optimization_summary(self, original_latex: str, optimized_latex: str, 
                               ats_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the optimizations made"""
        
        # Count keyword additions (basic implementation)
        missing_keywords = ats_analysis.get('missing', [])
        
        # Simple keyword counting in both versions
        original_lower = original_latex.lower()
        optimized_lower = optimized_latex.lower()
        
        keywords_added = []
        for keyword in missing_keywords:
            keyword_lower = keyword.lower()
            original_count = original_lower.count(keyword_lower)
            optimized_count = optimized_lower.count(keyword_lower)
            if optimized_count > original_count:
                keywords_added.append({
                    'keyword': keyword,
                    'original_count': original_count,
                    'optimized_count': optimized_count,
                    'added': optimized_count - original_count
                })
        
        return {
            'original_ats_score': ats_analysis.get('ats', 0),
            'keywords_successfully_added': len(keywords_added),
            'keyword_additions': keywords_added,
            'total_missing_keywords': len(missing_keywords),
            'optimization_coverage': len(keywords_added) / len(missing_keywords) * 100 if missing_keywords else 0
        }

# Example usage function
def optimize_resume_pipeline(ats_analysis_file: str, original_latex_file: str, 
                           resume_json_file: str, output_file: str, 
                           job_description: str = "") -> Dict[str, Any]:
    """
    Complete pipeline to optimize a resume
    
    Args:
        ats_analysis_file: Path to ATS analysis JSON file
        original_latex_file: Path to original LaTeX resume
        resume_json_file: Path to parsed resume JSON
        output_file: Path for optimized LaTeX output
        job_description: Optional job description
        
    Returns:
        Optimization summary dictionary
    """
    
    # Initialize optimizer
    optimizer = ResumeOptimizerAgent()
    
    try:
        # Load input files
        with open(ats_analysis_file, 'r', encoding='utf-8') as f:
            ats_analysis = json.load(f)
        
        with open(original_latex_file, 'r', encoding='utf-8') as f:
            original_latex = f.read()
        
        with open(resume_json_file, 'r', encoding='utf-8') as f:
            resume_json = json.load(f)
        
        # Optimize resume
        print("Optimizing resume...")
        optimized_latex = optimizer.optimize_resume(
            ats_analysis, original_latex, resume_json, job_description
        )
        
        # Save optimized resume
        success = optimizer.save_optimized_resume(optimized_latex, output_file)
        
        if success:
            # Generate optimization summary
            summary = optimizer.get_optimization_summary(
                original_latex, optimized_latex, ats_analysis
            )
            
            print(f"\nOptimization Summary:")
            print(f"Original ATS Score: {summary['original_ats_score']}")
            print(f"Keywords Added: {summary['keywords_successfully_added']}/{summary['total_missing_keywords']}")
            print(f"Optimization Coverage: {summary['optimization_coverage']:.1f}%")
            
            return summary
        else:
            return {"error": "Failed to save optimized resume"}
            
    except Exception as e:
        print(f"Error in optimization pipeline: {str(e)}")
        return {"error": str(e)}

# Command line interface
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize resume based on ATS analysis')
    parser.add_argument('--ats', required=True, help='Path to ATS analysis JSON file')
    parser.add_argument('--latex', required=True, help='Path to original LaTeX resume file')
    parser.add_argument('--resume', required=True, help='Path to parsed resume JSON file')
    parser.add_argument('--output', default='optimized_resume.tex', help='Output file path (default: optimized_resume.tex)')
    parser.add_argument('--job-desc', default='', help='Optional job description for better optimization')
    
    args = parser.parse_args()
    
    try:
        # Load files
        print(f"📂 Loading ATS analysis from: {args.ats}")
        with open(args.ats, 'r', encoding='utf-8') as f:
            ats_analysis = json.load(f)
        
        print(f"📂 Loading LaTeX file from: {args.latex}")
        with open(args.latex, 'r', encoding='utf-8') as f:
            original_latex = f.read()
        
        print(f"📂 Loading resume JSON from: {args.resume}")
        with open(args.resume, 'r', encoding='utf-8') as f:
            resume_json = json.load(f)
        
        # Initialize optimizer
        optimizer = ResumeOptimizerAgent()
        
        print("🔄 Optimizing resume...")
        
        # Run optimization
        optimized_latex = optimizer.optimize_resume(
            ats_analysis=ats_analysis,
            original_latex=original_latex, 
            resume_json=resume_json,
            job_description=args.job_desc
        )
        
        # Save to file
        with open(args.output, "w", encoding='utf-8') as f:
            f.write(optimized_latex)
        
        print(f"✅ Optimization complete!")
        print(f"📄 Optimized resume saved to: {args.output}")
        
        # Show optimization summary
        summary = optimizer.get_optimization_summary(original_latex, optimized_latex, ats_analysis)
        print(f"\n📊 Optimization Summary:")
        print(f"   Original ATS Score: {summary['original_ats_score']}")
        print(f"   Keywords Added: {summary['keywords_successfully_added']}/{summary['total_missing_keywords']}")
        print(f"   Coverage: {summary['optimization_coverage']:.1f}%")
        
        if summary['keyword_additions']:
            print(f"\n🎯 Keywords Successfully Added:")
            for kw in summary['keyword_additions'][:5]:  # Show first 5
                print(f"   • {kw['keyword']} (added {kw['added']} times)")
    
    except FileNotFoundError as e:
        print(f"❌ Error: File not found - {e}")
        print("Make sure all file paths are correct!")
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format - {e}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        
    # Show usage help if no arguments provided
    if len(sys.argv) == 1:
        print("\n💡 Usage Examples:")
        print("python resume_optimizer.py --ats ats_analysis.json --latex resume.tex --resume resume_data.json")
        print("python resume_optimizer.py --ats ats_analysis.json --latex resume.tex --resume resume_data.json --output my_optimized_resume.tex")
        print("python resume_optimizer.py --ats ats_analysis.json --latex resume.tex --resume resume_data.json --job-desc 'Software Engineer role requiring ML and systems'")
        parser.print_help()