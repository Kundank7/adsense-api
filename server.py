from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse, urljoin

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

app = FastAPI()
api_router = APIRouter(prefix="/api")

# Models
class WebsiteAnalyzeRequest(BaseModel):
    url: str

class AdminSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    openrouter_api_key: Optional[str] = None
    ai_enabled: bool = False
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class AdConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    position: str  # header, sidebar, inline, footer
    enabled: bool = True
    direction: str = "center"  # left, right, center
    size: str = "300x250"
    ad_code: str = ""
    frequency: Optional[int] = None  # for inline ads
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ScanResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    url: str
    overall_score: int
    adsense_score: int
    seo_score: int
    adsense_passed: bool
    results: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    scanned_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class AnalyticsUpdate(BaseModel):
    event_type: str  # visitor, scan
    metadata: Optional[Dict[str, Any]] = {}

# Helper Functions
async def fetch_url(url: str, timeout: int = 15) -> tuple[str, int, dict]:
    """Fetch URL and return content, status code, and headers"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout), allow_redirects=True) as response:
                content = await response.text()
                return content, response.status, dict(response.headers)
    except Exception as e:
        logging.error(f"Error fetching {url}: {str(e)}")
        return "", 0, {}

async def check_https(url: str) -> Dict[str, Any]:
    """Check if website uses HTTPS"""
    parsed = urlparse(url)
    return {
        "passed": parsed.scheme == "https",
        "message": "HTTPS enabled" if parsed.scheme == "https" else "HTTPS required for AdSense"
    }

async def check_content_quality(html: str, url: str) -> Dict[str, Any]:
    """Check content quality based on AdSense guidelines"""
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.decompose()
    
    text = soup.get_text(separator=' ', strip=True)
    word_count = len(text.split())
    
    # Check for original content indicators
    has_paragraphs = len(soup.find_all('p')) > 3
    has_headings = len(soup.find_all(['h1', 'h2', 'h3'])) > 0
    
    issues = []
    if word_count < 300:
        issues.append("Content too thin - minimum 300 words recommended")
    if not has_paragraphs:
        issues.append("Insufficient paragraph structure")
    if not has_headings:
        issues.append("Missing proper heading structure")
    
    score = 0
    if word_count >= 300:
        score += 30
    if word_count >= 600:
        score += 20
    if has_paragraphs:
        score += 25
    if has_headings:
        score += 25
    
    return {
        "score": min(score, 100),
        "word_count": word_count,
        "has_paragraphs": has_paragraphs,
        "has_headings": has_headings,
        "issues": issues,
        "passed": score >= 60
    }

async def check_prohibited_content(html: str) -> Dict[str, Any]:
    """Basic check for prohibited content keywords"""
    text = html.lower()
    
    prohibited_keywords = [
        'adult', 'porn', 'xxx', 'gambling', 'casino', 'viagra',
        'illegal', 'piracy', 'hacking', 'crack', 'keygen'
    ]
    
    found = [kw for kw in prohibited_keywords if kw in text]
    
    return {
        "passed": len(found) == 0,
        "issues": found if found else [],
        "message": "No prohibited content detected" if len(found) == 0 else f"Potential prohibited content: {', '.join(found)}"
    }

async def check_navigation(soup: BeautifulSoup, url: str) -> Dict[str, Any]:
    """Check for clear navigation structure"""
    nav_elements = soup.find_all(['nav', 'header'])
    links = soup.find_all('a', href=True)
    
    internal_links = [link for link in links if urlparse(link['href']).netloc in ['', urlparse(url).netloc]]
    
    has_nav = len(nav_elements) > 0
    has_menu = len(internal_links) > 3
    
    return {
        "passed": has_nav and has_menu,
        "has_navigation": has_nav,
        "internal_links_count": len(internal_links),
        "message": "Clear navigation present" if has_nav and has_menu else "Navigation structure needs improvement"
    }

async def check_required_pages(soup: BeautifulSoup, url: str) -> Dict[str, Any]:
    """Check for required pages: About, Contact, Privacy Policy"""
    links = soup.find_all('a', href=True)
    link_texts = [link.get_text().lower() for link in links]
    
    has_about = any('about' in text for text in link_texts)
    has_contact = any('contact' in text for text in link_texts)
    has_privacy = any('privacy' in text for text in link_texts)
    
    missing = []
    if not has_about:
        missing.append("About page")
    if not has_contact:
        missing.append("Contact page")
    if not has_privacy:
        missing.append("Privacy Policy")
    
    return {
        "has_about": has_about,
        "has_contact": has_contact,
        "has_privacy": has_privacy,
        "missing_pages": missing,
        "passed": len(missing) == 0,
        "message": "All required pages present" if len(missing) == 0 else f"Missing: {', '.join(missing)}"
    }

async def check_mobile_friendly(html: str) -> Dict[str, Any]:
    """Check for mobile-friendly indicators"""
    soup = BeautifulSoup(html, 'html.parser')
    
    viewport_meta = soup.find('meta', attrs={'name': 'viewport'})
    has_viewport = viewport_meta is not None
    
    # Check for responsive indicators
    has_media_queries = '@media' in html
    
    return {
        "passed": has_viewport,
        "has_viewport_meta": has_viewport,
        "has_media_queries": has_media_queries,
        "message": "Mobile-friendly" if has_viewport else "Viewport meta tag missing"
    }

async def check_seo_basics(soup: BeautifulSoup, url: str) -> Dict[str, Any]:
    """Check basic SEO elements"""
    title = soup.find('title')
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    h1_tags = soup.find_all('h1')
    
    img_tags = soup.find_all('img')
    imgs_with_alt = [img for img in img_tags if img.get('alt')]
    
    issues = []
    score = 0
    
    if title and len(title.get_text()) > 10:
        score += 25
    else:
        issues.append("Missing or short title tag")
    
    if meta_desc and len(meta_desc.get('content', '')) > 50:
        score += 25
    else:
        issues.append("Missing or short meta description")
    
    if len(h1_tags) == 1:
        score += 25
    elif len(h1_tags) == 0:
        issues.append("No H1 tag found")
    else:
        issues.append("Multiple H1 tags found")
    
    if img_tags and len(imgs_with_alt) / len(img_tags) > 0.7:
        score += 25
    elif img_tags:
        issues.append("Many images missing alt attributes")
    
    return {
        "score": score,
        "has_title": title is not None,
        "has_meta_description": meta_desc is not None,
        "h1_count": len(h1_tags),
        "images_with_alt_percentage": round(len(imgs_with_alt) / len(img_tags) * 100) if img_tags else 100,
        "issues": issues,
        "passed": score >= 50
    }

async def get_pagespeed_insights(url: str) -> Dict[str, Any]:
    """Get PageSpeed Insights data (simplified without API key)"""
    # Since we're using free API, we'll do basic performance checks
    try:
        start_time = asyncio.get_event_loop().time()
        html, status, headers = await fetch_url(url)
        load_time = asyncio.get_event_loop().time() - start_time
        
        score = 100
        if load_time > 3:
            score = 50
        elif load_time > 1.5:
            score = 70
        elif load_time > 1:
            score = 85
        
        return {
            "score": int(score),
            "load_time": round(load_time, 2),
            "passed": score >= 60,
            "message": f"Load time: {round(load_time, 2)}s"
        }
    except Exception as e:
        return {
            "score": 0,
            "load_time": 0,
            "passed": False,
            "message": f"Could not measure: {str(e)}"
        }

async def analyze_website(url: str) -> Dict[str, Any]:
    """Main analysis function"""
    # Ensure URL has scheme
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    # Fetch website
    html, status, headers = await fetch_url(url)
    
    if status != 200 or not html:
        raise HTTPException(status_code=400, detail="Could not fetch website. Please check the URL.")
    
    soup = BeautifulSoup(html, 'html.parser')
    
    # Run all checks
    https_check = await check_https(url)
    content_check = await check_content_quality(html, url)
    prohibited_check = await check_prohibited_content(html)
    navigation_check = await check_navigation(soup, url)
    required_pages_check = await check_required_pages(soup, url)
    mobile_check = await check_mobile_friendly(html)
    seo_check = await check_seo_basics(soup, url)
    performance_check = await get_pagespeed_insights(url)
    
    # Calculate AdSense score
    adsense_checks = [
        https_check['passed'],
        content_check['passed'],
        prohibited_check['passed'],
        navigation_check['passed'],
        required_pages_check['passed'],
        mobile_check['passed']
    ]
    adsense_score = int((sum(adsense_checks) / len(adsense_checks)) * 100)
    adsense_passed = adsense_score >= 80
    
    # Calculate SEO score
    seo_score = int((seo_check['score'] + performance_check['score']) / 2)
    
    # Overall score
    overall_score = int((adsense_score + seo_score) / 2)
    
    # Generate recommendations
    recommendations = []
    
    if not https_check['passed']:
        recommendations.append({"priority": "high", "category": "Security", "issue": "Enable HTTPS", "description": "AdSense requires HTTPS for approval"})
    
    if not content_check['passed']:
        recommendations.append({"priority": "high", "category": "Content", "issue": "Improve content quality", "description": f"Add more original content. Current: {content_check['word_count']} words"})
    
    if not prohibited_check['passed']:
        recommendations.append({"priority": "high", "category": "Policy", "issue": "Prohibited content detected", "description": "Remove content that violates AdSense policies"})
    
    if not required_pages_check['passed']:
        recommendations.append({"priority": "high", "category": "Pages", "issue": "Add required pages", "description": f"Missing: {', '.join(required_pages_check['missing_pages'])}"})
    
    if not mobile_check['passed']:
        recommendations.append({"priority": "medium", "category": "Technical", "issue": "Mobile optimization", "description": "Add viewport meta tag for mobile-friendly display"})
    
    if not navigation_check['passed']:
        recommendations.append({"priority": "medium", "category": "UX", "issue": "Improve navigation", "description": "Add clear navigation menu with internal links"})
    
    if seo_check['issues']:
        for issue in seo_check['issues']:
            recommendations.append({"priority": "medium", "category": "SEO", "issue": issue, "description": "Fix to improve search visibility"})
    
    if performance_check['score'] < 60:
        recommendations.append({"priority": "medium", "category": "Performance", "issue": "Slow loading speed", "description": f"Current load time: {performance_check['load_time']}s. Optimize for faster loading"})
    
    results = {
        "adsense_checks": {
            "https": https_check,
            "content_quality": content_check,
            "prohibited_content": prohibited_check,
            "navigation": navigation_check,
            "required_pages": required_pages_check,
            "mobile_friendly": mobile_check
        },
        "seo_checks": {
            "basics": seo_check,
            "performance": performance_check
        }
    }
    
    return {
        "overall_score": overall_score,
        "adsense_score": adsense_score,
        "seo_score": seo_score,
        "adsense_passed": adsense_passed,
        "results": results,
        "recommendations": recommendations
    }

# API Routes
@api_router.get("/")
async def root():
    return {"message": "AdSense Ready Pro API"}

@api_router.post("/analyze")
async def analyze_site(request: WebsiteAnalyzeRequest):
    """Analyze website for AdSense readiness"""
    try:
        analysis = await analyze_website(request.url)
        
        # Save to database
        scan_result = ScanResult(
            url=request.url,
            overall_score=analysis['overall_score'],
            adsense_score=analysis['adsense_score'],
            seo_score=analysis['seo_score'],
            adsense_passed=analysis['adsense_passed'],
            results=analysis['results'],
            recommendations=analysis['recommendations']
        )
        
        doc = scan_result.model_dump()
        doc['scanned_at'] = doc['scanned_at'].isoformat()
        await db.scans.insert_one(doc)
        
        return analysis
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@api_router.get("/scans", response_model=List[ScanResult])
async def get_scans(limit: int = 50):
    """Get recent scans"""
    scans = await db.scans.find({}, {"_id": 0}).sort("scanned_at", -1).to_list(limit)
    for scan in scans:
        if isinstance(scan['scanned_at'], str):
            scan['scanned_at'] = datetime.fromisoformat(scan['scanned_at'])
    return scans

@api_router.get("/admin/settings")
async def get_admin_settings():
    """Get admin settings"""
    settings = await db.admin_settings.find_one({}, {"_id": 0})
    if not settings:
        default_settings = AdminSettings()
        doc = default_settings.model_dump()
        doc['updated_at'] = doc['updated_at'].isoformat()
        await db.admin_settings.insert_one(doc)
        return default_settings.model_dump()
    return settings

@api_router.post("/admin/settings")
async def update_admin_settings(settings: AdminSettings):
    """Update admin settings"""
    doc = settings.model_dump()
    doc['updated_at'] = datetime.now(timezone.utc).isoformat()
    await db.admin_settings.delete_many({})
    await db.admin_settings.insert_one(doc)
    return {"success": True, "message": "Settings updated"}

@api_router.get("/admin/ads", response_model=List[AdConfig])
async def get_ad_configs():
    """Get all ad configurations"""
    ads = await db.ad_configs.find({}, {"_id": 0}).to_list(100)
    for ad in ads:
        if isinstance(ad.get('created_at'), str):
            ad['created_at'] = datetime.fromisoformat(ad['created_at'])
    return ads

@api_router.post("/admin/ads")
async def create_ad_config(ad: AdConfig):
    """Create new ad configuration"""
    doc = ad.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    result = await db.ad_configs.insert_one(doc)
    # Remove the MongoDB _id from the response
    doc.pop('_id', None)
    return {"success": True, "ad": doc}

@api_router.put("/admin/ads/{ad_id}")
async def update_ad_config(ad_id: str, ad: AdConfig):
    """Update ad configuration"""
    doc = ad.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    await db.ad_configs.update_one({"id": ad_id}, {"$set": doc})
    return {"success": True}

@api_router.delete("/admin/ads/{ad_id}")
async def delete_ad_config(ad_id: str):
    """Delete ad configuration"""
    await db.ad_configs.delete_one({"id": ad_id})
    return {"success": True}

@api_router.get("/admin/analytics")
async def get_analytics():
    """Get analytics data"""
    total_scans = await db.scans.count_documents({})
    
    # Get most scanned websites
    pipeline = [
        {"$group": {"_id": "$url", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10}
    ]
    top_sites = await db.scans.aggregate(pipeline).to_list(10)
    
    # Get recent activity
    recent_scans = await db.scans.find({}, {"_id": 0, "url": 1, "overall_score": 1, "scanned_at": 1}).sort("scanned_at", -1).limit(10).to_list(10)
    
    return {
        "total_scans": total_scans,
        "top_scanned_sites": top_sites,
        "recent_activity": recent_scans
    }

@api_router.post("/analytics")
async def track_analytics(data: AnalyticsUpdate):
    """Track analytics events"""
    event = {
        "event_type": data.event_type,
        "metadata": data.metadata,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    await db.analytics.insert_one(event)
    return {"success": True}

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()