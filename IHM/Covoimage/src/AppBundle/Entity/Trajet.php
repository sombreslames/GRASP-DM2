<?php
namespace AppBundle\Entity;
use Symfony\Component\Validator\Constraints as CSTRT;
use Doctrine\ORM\Mapping as ORM;

/**
 * @ORM\Entity
 * @ORM\Table(name="Trajet")
 */
class Trajet
{
	/**
	 * Clé primaire de Trajet
	 * @ORM\Column(name="ID_trajet", type="integer")
	 * @ORM\Id
	 * @ORM\GeneratedValue(strategy="AUTO")
	 */
	private $ID_trajet;
	/**
	 * @ORM\Column(name="nbplaces", type="integer")
	 */
	private $nbplaces;
	/**
	 * Un personne est conducteur par trajet
	 * Une personne peut être conducteur pour plusieurs trajet
	 * @ORM\Column(name="ID_conducteur", type="integer")
	 * @ORM\ManyToOne(targetEntity="Personne")
     * @ORM\JoinColumn(name="ID_conducteur", referencedColumnName="ID_personne")
	 */
	private $ID_conducteur;
	/**
	 * Un trajet utilise un seul vehicule
	 * Un vehicule peut etre utlisé pour plusieurs trajets
	 * @ORM\Column(name="ID_vehicule", type="integer")
	 * @ORM\ManyToOne(targetEntity="Vehicule")
     * @ORM\JoinColumn(name="ID_vehicule", referencedColumnName="ID_vehicule")
	 */
	private $ID_vehicule;
	/**
	 * @ORM\Column(name="description", type="string")
	 */
	private $description;
    /**
     * @ORM\Column(name="villedep", type="string")
     *
     *
     */
    private $villedep;
    /**
     * @ORM\Column(name="ID_villedep", type="string")
     *
     */
    private $ID_villedep;
    /**
     * @ORM\Column(name="villefin", type="string")
     *
     *
     */
    private $villefin;
    /**
     * @ORM\Column(name="ID_villefin", type="string")
     *
     */
    private $ID_villefin;
    /**
     * Get iDTrajet
     *
     * @return integer
     */

    public function getIDTrajet()
    {
        return $this->ID_trajet;
    }

    /**
     * Set nbplaces
     *
     * @param integer $nbplaces
     *
     * @return Trajet
     */
    public function setNbplaces($nbplaces)
    {
        $this->nbplaces = $nbplaces;

        return $this;
    }

    /**
     * Get nbplaces
     *
     * @return integer
     */
    public function getNbplaces()
    {
        return $this->nbplaces;
    }

    /**
     * Set iDConducteur
     *
     * @param integer $iDConducteur
     *
     * @return Trajet
     */
    public function setIDConducteur($iDConducteur)
    {
        $this->ID_conducteur = $iDConducteur;

        return $this;
    }

    /**
     * Get iDConducteur
     *
     * @return integer
     */
    public function getIDConducteur()
    {
        return $this->ID_conducteur;
    }

    /**
     * Set iDVehicule
     *
     * @param integer $iDVehicule
     *
     * @return Trajet
     */
    public function setIDVehicule($idvehicule)
    {
        $this->ID_vehicule = $idvehicule;

        return $this;
    }

    /**
     * Get iDVehicule
     *
     * @return integer
     */
    public function getIDVehicule()
    {
        return $this->id_vehicule;
    }

    /**
     * Set description
     *
     * @param string $description
     *
     * @return Trajet
     */
    public function setDescription($description)
    {
        $this->description = $description;

        return $this;
    }

    /**
     * Get description
     *
     * @return string
     */
    public function getDescription()
    {
        return $this->description;
    }

    /**
     * Set villedep
     *
     * @param string $villedep
     *
     * @return Trajet
     */
    public function setVilledep($villedep)
    {
        $this->villedep = $villedep;

        return $this;
    }

    /**
     * Get villedep
     *
     * @return string
     */
    public function getVilledep()
    {
        return $this->villedep;
    }

    /**
     * Set iDVilledep
     *
     * @param string $iDVilledep
     *
     * @return Trajet
     */
    public function setIDVilledep($iDVilledep)
    {
        $this->ID_villedep = $iDVilledep;

        return $this;
    }

    /**
     * Get iDVilledep
     *
     * @return string
     */
    public function getIDVilledep()
    {
        return $this->ID_villedep;
    }

    /**
     * Set villefin
     *
     * @param string $villefin
     *
     * @return Trajet
     */
    public function setVillefin($villefin)
    {
        $this->villefin = $villefin;

        return $this;
    }

    /**
     * Get villefin
     *
     * @return string
     */
    public function getVillefin()
    {
        return $this->villefin;
    }

    /**
     * Set iDVillefin
     *
     * @param string $iDVillefin
     *
     * @return Trajet
     */
    public function setIDVillefin($iDVillefin)
    {
        $this->ID_villefin = $iDVillefin;

        return $this;
    }

    /**
     * Get iDVillefin
     *
     * @return string
     */
    public function getIDVillefin()
    {
        return $this->ID_villefin;
    }
}
