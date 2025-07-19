import fitz  # PyMuPDF
import re
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from difflib import get_close_matches
import sys
import os
from pathlib import Path

class ResumeParser:
    def __init__(self):
        self.section_keywords = {
            'contact': ['email', 'phone', 'address', 'linkedin', 'github'],
            'summary': ['summary', 'objective', 'profile', 'about'],
            'experience': ['experience', 'work history', 'employment', 'professional experience'],
            'education': ['education', 'academic', 'university', 'college', 'degree'],
            'skills': ['skills', 'technologies', 'technical skills', 'competencies', 'tools'],
            'projects': ['projects', 'personal projects', 'key projects'],
            'certifications': ['certifications', 'certificates', 'licenses']
        }
        self.skill_keywords = self._load_skills()
        
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def _load_skills(self) -> List[str]:
        """Load predefined skill keywords for extraction."""
        return [
            'python', 'java', 'javascript', 'typescript', 'react', 'node.js', 'nodejs',
            'angular', 'vue', 'html', 'css', 'sql', 'mongodb', 'postgresql', 'mysql',
            'aws', 'docker', 'kubernetes', 'git', 'linux', 'windows', 'macos',
            'machine learning', 'data science', 'artificial intelligence', 'ai',
            'rest api', 'graphql', 'microservices', 'agile', 'scrum', 'devops',
            'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn',
            'c++', 'c#', 'go', 'rust', 'swift', 'kotlin', 'php', 'ruby', 'flask',
            'django', 'spring boot', 'express', 'bootstrap', 'tailwind', 'redis',
            'elasticsearch', 'jenkins', 'terraform', 'ansible', 'prometheus'
        ]

    def extract_text_from_pdf(self, pdf_path: str, extract_images: bool = False) -> str:
        """
        Extract text from PDF using PyMuPDF with enhanced options.
        
        Args:
            pdf_path: Path to the PDF file
            extract_images: Whether to attempt OCR on images (basic implementation)
            
        Returns:
            Cleaned text content from the PDF
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not pdf_path.suffix.lower() == '.pdf':
            raise ValueError("File must have .pdf extension")
            
        try:
            # Open PDF document
            doc = fitz.open(str(pdf_path))
            self.logger.info(f"Successfully opened PDF: {pdf_path.name} ({doc.page_count} pages)")
            
            extracted_text = []
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Extract text with different methods for better coverage
                text = page.get_text("text")  # Plain text
                
                # If no text found and extract_images is True, try to get text from images
                if not text.strip() and extract_images:
                    text = self._extract_text_from_images(page)
                
                if text.strip():
                    extracted_text.append(text)
                    self.logger.debug(f"Extracted {len(text)} characters from page {page_num + 1}")
                else:
                    self.logger.warning(f"No text found on page {page_num + 1}")
            
            doc.close()
            
            # Combine all pages
            full_text = "\n".join(extracted_text)
            cleaned_text = self._clean_text(full_text)
            
            self.logger.info(f"Text extraction completed. Total characters: {len(cleaned_text)}")
            return cleaned_text
            
        except Exception as e:
            self.logger.error(f"PDF extraction failed for {pdf_path}: {str(e)}")
            raise

    def _extract_text_from_images(self, page: fitz.Page) -> str:
        """
        Basic image text extraction (placeholder for OCR functionality).
        For production use, integrate with Tesseract or similar OCR library.
        """
        # This is a placeholder - in production, you'd use OCR libraries
        # like pytesseract with the image data from PyMuPDF
        try:
            image_list = page.get_images(full=True)
            if image_list:
                self.logger.info(f"Found {len(image_list)} images on page, OCR not implemented")
            return ""
        except Exception as e:
            self.logger.warning(f"Image extraction failed: {e}")
            return ""

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace and normalize line breaks
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        
        # Replace various unicode spaces with regular spaces
        text = re.sub(r'[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000]', ' ', text)
        
        # Clean up special characters but preserve essential punctuation
        text = re.sub(r'[^\w\s@.\-+()/#&,:\'\"°%$]', ' ', text)
        
        # Normalize multiple spaces
        text = re.sub(r' +', ' ', text)
        
        return text.strip()

    def get_document_metadata(self, pdf_path: str) -> Dict[str, str]:
        """Extract metadata from the PDF document."""
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            doc.close()
            
            return {
                'title': metadata.get('title', ''),
                'author': metadata.get('author', ''),
                'subject': metadata.get('subject', ''),
                'creator': metadata.get('creator', ''),
                'producer': metadata.get('producer', ''),
                'creation_date': metadata.get('creationDate', ''),
                'modification_date': metadata.get('modDate', '')
            }
        except Exception as e:
            self.logger.error(f"Failed to extract metadata: {e}")
            return {}

    def parse_resume_sections(self, text: str) -> Dict[str, str]:
        """Parse resume into distinct sections."""
        sections = {}
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if not lines:
            self.logger.warning("No text lines found for section parsing")
            return sections
            
        boundaries = self._find_section_boundaries(lines)
        self.logger.info(f"Found section boundaries: {list(boundaries.keys())}")

        for section_name in self.section_keywords:
            start = boundaries.get(section_name)
            if start is not None:
                # Find the next section boundary or end of document
                next_boundaries = [idx for name, idx in boundaries.items() if idx > start]
                end = min(next_boundaries) if next_boundaries else len(lines)
                
                content = "\n".join(lines[start + 1:end]).strip()
                if content:
                    sections[section_name] = content
                    self.logger.debug(f"Extracted {section_name} section ({len(content)} chars)")

        return sections

    def _find_section_boundaries(self, lines: List[str]) -> Dict[str, int]:
        """Identify section headers in the resume text."""
        boundaries = {}
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Skip very short lines or lines with mostly punctuation
            if len(line_lower) < 3 or len(re.sub(r'[^\w\s]', '', line_lower)) < 2:
                continue
                
            for section, keywords in self.section_keywords.items():
                if section in boundaries:  # Already found this section
                    continue
                    
                # Direct match
                if any(keyword in line_lower for keyword in keywords):
                    boundaries[section] = i
                    continue
                
                # Fuzzy matching for misspellings
                matches = get_close_matches(line_lower, keywords, n=1, cutoff=0.8)
                if matches:
                    boundaries[section] = i
                    
        return boundaries

    def extract_contact_info(self, text: str) -> Dict[str, str]:
        """Extract contact information using improved regex patterns."""
        contact_info = {}

        # Email with better pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        if email_match:
            contact_info['email'] = email_match.group()

        # Phone with international support
        phone_pattern = r'(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}|\+\d{1,3}[-.\s]?\d{1,14}'
        phone_match = re.search(phone_pattern, text)
        if phone_match:
            contact_info['phone'] = phone_match.group().strip()

        # LinkedIn profile
        linkedin_pattern = r'(?:https?://)?(?:www\.)?linkedin\.com/in/[^\s\)>\"\']+/?'
        linkedin_match = re.search(linkedin_pattern, text, re.IGNORECASE)
        if linkedin_match:
            contact_info['linkedin'] = linkedin_match.group()

        # GitHub profile
        github_pattern = r'(?:https?://)?(?:www\.)?github\.com/[^\s\)>\"\']+/?'
        github_match = re.search(github_pattern, text, re.IGNORECASE)
        if github_match:
            contact_info['github'] = github_match.group()

        # Basic address extraction (city, state)
        address_pattern = r'\b[A-Z][a-z]+,\s*[A-Z]{2}\b|\b[A-Z][a-z\s]+,\s*[A-Z][a-z\s]+\b'
        address_match = re.search(address_pattern, text)
        if address_match:
            contact_info['address'] = address_match.group()

        return contact_info

    def extract_skills(self, text: str, include_context: bool = False) -> List[str] | Dict[str, List[str]]:
        """
        Extract skills from text with optional context.
        
        Args:
            text: Resume text
            include_context: If True, return dict with skill categories
            
        Returns:
            List of skills or categorized dict if include_context=True
        """
        text_lower = text.lower()
        found_skills = []
        
        for skill in self.skill_keywords:
            # Use word boundaries for better matching
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.append(skill.title())
        
        # Remove duplicates and sort
        found_skills = sorted(list(set(found_skills)))
        
        if not include_context:
            return found_skills
        
        # Categorize skills (basic implementation)
        categories = {
            'Programming Languages': [],
            'Frameworks/Libraries': [],
            'Databases': [],
            'Tools/Platforms': [],
            'Other': []
        }
        
        for skill in found_skills:
            skill_lower = skill.lower()
            if skill_lower in ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust', 'swift', 'kotlin', 'php', 'ruby']:
                categories['Programming Languages'].append(skill)
            elif skill_lower in ['react', 'angular', 'vue', 'flask', 'django', 'spring boot', 'express']:
                categories['Frameworks/Libraries'].append(skill)
            elif skill_lower in ['sql', 'mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch']:
                categories['Databases'].append(skill)
            elif skill_lower in ['aws', 'docker', 'kubernetes', 'git', 'jenkins', 'terraform']:
                categories['Tools/Platforms'].append(skill)
            else:
                categories['Other'].append(skill)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}

    def get_resume_stats(self, text: str, pdf_path: Optional[str] = None) -> Dict[str, any]:
        """Generate comprehensive resume statistics."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        words = text.split()
        sections = self.parse_resume_sections(text)
        
        stats = {
            "file_info": {},
            "text_stats": {
                "total_lines": len(lines),
                "total_words": len(words),
                "total_characters": len(text),
                "average_words_per_line": round(len(words) / len(lines) if lines else 0, 2)
            },
            "content_analysis": {
                "sections_found": len(sections),
                "section_names": list(sections.keys()),
                "skills_count": len(self.extract_skills(text)),
                "has_contact_info": bool(self.extract_contact_info(text))
            }
        }
        
        if pdf_path:
            try:
                file_path = Path(pdf_path)
                stats["file_info"] = {
                    "filename": file_path.name,
                    "file_size_kb": round(file_path.stat().st_size / 1024, 2),
                }
                
                # Add metadata if available
                metadata = self.get_document_metadata(pdf_path)
                if metadata:
                    stats["file_info"]["metadata"] = metadata
                    
            except Exception as e:
                self.logger.warning(f"Could not get file info: {e}")
        
        return stats

    def analyze_resume_completeness(self, text: str) -> Dict[str, any]:
        """Analyze resume completeness and provide suggestions."""
        sections = self.parse_resume_sections(text)
        contact_info = self.extract_contact_info(text)
        skills = self.extract_skills(text)
        
        essential_sections = ['contact', 'experience', 'education', 'skills']
        optional_sections = ['summary', 'projects', 'certifications']
        
        missing_essential = [s for s in essential_sections if s not in sections]
        missing_optional = [s for s in optional_sections if s not in sections]
        
        completeness_score = ((len(essential_sections) - len(missing_essential)) / len(essential_sections)) * 100
        
        return {
            "completeness_score": round(completeness_score, 1),
            "sections_present": list(sections.keys()),
            "missing_essential": missing_essential,
            "missing_optional": missing_optional,
            "contact_fields_found": len(contact_info),
            "skills_found": len(skills),
            "recommendations": self._generate_recommendations(missing_essential, missing_optional, contact_info, skills)
        }

    def _generate_recommendations(self, missing_essential: List[str], missing_optional: List[str], 
                                contact_info: Dict, skills: List[str]) -> List[str]:
        """Generate improvement recommendations for the resume."""
        recommendations = []
        
        if missing_essential:
            recommendations.append(f"Add missing essential sections: {', '.join(missing_essential)}")
        
        if len(contact_info) < 2:
            recommendations.append("Include more contact information (email, phone, LinkedIn)")
        
        if len(skills) < 5:
            recommendations.append("Consider adding more relevant skills to strengthen your profile")
        
        if missing_optional:
            recommendations.append(f"Consider adding optional sections: {', '.join(missing_optional[:2])}")
        
    def save_resume_metadata(self, pdf_path: str, output_path: Optional[str] = None) -> str:
        """
        Extract all resume data and save to a JSON metadata file.
        
        Args:
            pdf_path: Path to the PDF resume file
            output_path: Optional custom output path for JSON file
            
        Returns:
            Path to the created JSON file
        """
        try:
            # Extract text and analyze
            resume_text = self.extract_text_from_pdf(pdf_path)
            pdf_file_path = Path(pdf_path)
            
            # Create comprehensive metadata
            metadata = {
                "file_info": {
                    "original_filename": pdf_file_path.name,
                    "file_path": str(pdf_file_path.absolute()),
                    "file_size_bytes": pdf_file_path.stat().st_size,
                    "file_size_kb": round(pdf_file_path.stat().st_size / 1024, 2),
                    "processed_date": datetime.now().isoformat(),
                    "pdf_metadata": self.get_document_metadata(pdf_path)
                },
                "contact_information": self.extract_contact_info(resume_text),
                "sections": self.parse_resume_sections(resume_text),
                "skills": {
                    "all_skills": self.extract_skills(resume_text),
                    "categorized_skills": self.extract_skills(resume_text, include_context=True),
                    "total_skills_count": len(self.extract_skills(resume_text))
                },
                "text_statistics": self.get_resume_stats(resume_text, pdf_path),
                "completeness_analysis": self.analyze_resume_completeness(resume_text),
                "raw_text": resume_text,  # Include full text for future processing
                "extraction_metadata": {
                    "parser_version": "2.0",
                    "pymupdf_version": fitz.__version__ if hasattr(fitz, '__version__') else "unknown",
                    "total_pages_processed": len(resume_text.split('\f')) if '\f' in resume_text else 1
                }
            }
            
            # Determine output path
            if output_path is None:
                output_path = pdf_file_path.parent / f"{pdf_file_path.stem}_metadata.json"
            else:
                output_path = Path(output_path)
                if output_path.is_dir():
                    output_path = output_path / f"{pdf_file_path.stem}_metadata.json"
            
            # Save to JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Metadata saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")
            raise

    def load_resume_metadata(self, json_path: str) -> Dict:
        """Load previously saved resume metadata from JSON file."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            self.logger.info(f"Metadata loaded from: {json_path}")
            return metadata
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
            raise

    def update_skills_database(self, metadata_files: List[str], output_path: str = "skills_database.json"):
        """
        Aggregate skills from multiple resume metadata files to build a skills database.
        
        Args:
            metadata_files: List of paths to resume metadata JSON files
            output_path: Path to save the aggregated skills database
        """
        skills_db = {
            "aggregation_info": {
                "created_date": datetime.now().isoformat(),
                "total_resumes_processed": len(metadata_files),
                "source_files": metadata_files
            },
            "skill_frequency": {},
            "skill_categories": {},
            "unique_skills": set(),
            "resume_sources": []
        }
        
        for metadata_file in metadata_files:
            try:
                metadata = self.load_resume_metadata(metadata_file)
                resume_info = {
                    "source_file": metadata_file,
                    "original_pdf": metadata.get("file_info", {}).get("original_filename", "unknown"),
                    "skills_count": metadata.get("skills", {}).get("total_skills_count", 0),
                    "skills": metadata.get("skills", {}).get("all_skills", [])
                }
                skills_db["resume_sources"].append(resume_info)
                
                # Aggregate skills frequency
                for skill in metadata.get("skills", {}).get("all_skills", []):
                    skills_db["skill_frequency"][skill] = skills_db["skill_frequency"].get(skill, 0) + 1
                    skills_db["unique_skills"].add(skill)
                
                # Aggregate categorized skills
                categorized = metadata.get("skills", {}).get("categorized_skills", {})
                for category, skill_list in categorized.items():
                    if category not in skills_db["skill_categories"]:
                        skills_db["skill_categories"][category] = {}
                    for skill in skill_list:
                        skills_db["skill_categories"][category][skill] = skills_db["skill_categories"][category].get(skill, 0) + 1
                        
            except Exception as e:
                self.logger.error(f"Error processing {metadata_file}: {e}")
                continue
        
        # Convert set to sorted list for JSON serialization
        skills_db["unique_skills"] = sorted(list(skills_db["unique_skills"]))
        skills_db["total_unique_skills"] = len(skills_db["unique_skills"])
        
        # Add top skills statistics
        skills_db["top_skills"] = sorted(
            skills_db["skill_frequency"].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:20]  # Top 20 most frequent skills
        
        # Save skills database
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(skills_db, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Skills database saved to: {output_path}")
        return output_path


# 🚀 Enhanced CLI Entry Point
if __name__ == "__main__":
    if len(sys.argv) not in [2, 3, 4]:
        print("Usage:")
        print("  python resume_parser.py <resume.pdf> [--detailed] [--save-json]")
        print("  python resume_parser.py --build-skills-db <metadata_dir>")
        print("")
        print("Options:")
        print("  --detailed     : Show detailed analysis and recommendations")
        print("  --save-json    : Save all extracted data to JSON metadata file")
        print("  --build-skills-db : Aggregate skills from all metadata files in directory")
        sys.exit(1)

    # Check for skills database building mode
    if "--build-skills-db" in sys.argv:
        metadata_dir = sys.argv[2]
        if not os.path.isdir(metadata_dir):
            print(f"❌ Error: Directory not found: {metadata_dir}")
            sys.exit(1)
        
        parser = ResumeParser()
        metadata_files = [
            os.path.join(metadata_dir, f) 
            for f in os.listdir(metadata_dir) 
            if f.endswith('_metadata.json')
        ]
        
        if not metadata_files:
            print(f"❌ Error: No *_metadata.json files found in {metadata_dir}")
            sys.exit(1)
        
        print(f"🔨 Building skills database from {len(metadata_files)} metadata files...")
        db_path = parser.update_skills_database(metadata_files)
        print(f"✅ Skills database created: {db_path}")
        sys.exit(0)

    # Normal resume processing mode
    pdf_path = sys.argv[1]
    detailed = "--detailed" in sys.argv
    save_json = "--save-json" in sys.argv

    # Debug: Print the path being checked
    print(f"🔍 Checking path: '{pdf_path}'")
    print(f"   Path exists: {os.path.exists(pdf_path)}")
    print(f"   Is PDF: {pdf_path.lower().endswith('.pdf')}")
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: File not found: {pdf_path}")
        print(f"   Current working directory: {os.getcwd()}")
        print(f"   Files in current directory: {os.listdir('.')}")
        sys.exit(1)
    
    if not pdf_path.lower().endswith(".pdf"):
        print(f"❌ Error: File must have .pdf extension: {pdf_path}")
        sys.exit(1)

    try:
        parser = ResumeParser()
        print(f"🔍 Analyzing resume: {Path(pdf_path).name}")
        
        # Save JSON metadata if requested
        json_path = None
        if save_json:
            print("💾 Saving metadata to JSON...")
            json_path = parser.save_resume_metadata(pdf_path)
            print(f"✅ Metadata saved to: {json_path}")
        
        # Extract text for display
        resume_text = parser.extract_text_from_pdf(pdf_path)
        
        if not resume_text.strip():
            print("⚠️  Warning: No text could be extracted from the PDF.")
            sys.exit(1)
        
        # Basic analysis
        print("\n📑 Contact Information:")
        contact_info = parser.extract_contact_info(resume_text)
        if contact_info:
            for key, value in contact_info.items():
                print(f"  {key.title()}: {value}")
        else:
            print("  No contact information found")
        
        print("\n📋 Sections Found:")
        sections = parser.parse_resume_sections(resume_text)
        if sections:
            for section_name in sections:
                print(f"  ✓ {section_name.replace('_', ' ').title()}")
        else:
            print("  No standard sections detected")
        
        print("\n🛠️  Skills Extracted:")
        if detailed:
            skills = parser.extract_skills(resume_text, include_context=True)
            if skills:
                for category, skill_list in skills.items():
                    print(f"  {category}: {', '.join(skill_list)}")
            else:
                print("  No skills detected")
        else:
            skills = parser.extract_skills(resume_text)
            if skills:
                print(f"  Found {len(skills)} skills: {', '.join(skills)}")
            else:
                print("  No skills detected")
        
        print("\n📊 Resume Statistics:")
        stats = parser.get_resume_stats(resume_text, pdf_path)
        print(f"  Lines: {stats['text_stats']['total_lines']}")
        print(f"  Words: {stats['text_stats']['total_words']}")
        print(f"  Characters: {stats['text_stats']['total_characters']}")
        print(f"  Sections: {stats['content_analysis']['sections_found']}")
        
        if detailed:
            print("\n🎯 Completeness Analysis:")
            analysis = parser.analyze_resume_completeness(resume_text)
            print(f"  Completeness Score: {analysis['completeness_score']}%")
            
            if analysis['missing_essential']:
                print(f"  ⚠️  Missing Essential: {', '.join(analysis['missing_essential'])}")
            
            if analysis['recommendations']:
                print("\n💡 Recommendations:")
                for rec in analysis['recommendations']:
                    print(f"  • {rec}")
        
        if json_path:
            print(f"\n📄 Metadata File: {json_path}")
            print("   Use this file for building skills databases or further analysis")
        
        print(f"\n✅ Analysis complete! Processed {len(resume_text)} characters from {Path(pdf_path).name}")
        
    except Exception as e:
        print(f"❌ Error processing resume: {str(e)}")
        sys.exit(1)